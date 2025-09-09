# 🔐 GCP IAM: Identity and Access Management

> **Master GCP IAM for secure access control and identity management**

## 📚 Concept

Google Cloud Identity and Access Management (IAM) lets administrators authorize who can take action on specific resources, giving you full control and visibility to manage cloud resources centrally. IAM provides a unified view of security policy across your entire organization.

### Key Features
- **Fine-grained Access Control**: Granular permissions for resources
- **Role-based Access**: Predefined and custom roles
- **Service Accounts**: Identity for applications and services
- **Organization Policies**: Centralized policy management
- **Audit Logging**: Comprehensive access logging
- **Federation**: External identity provider integration

## 🏗️ IAM Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GCP IAM Architecture                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Users     │  │   Groups    │  │   Service   │     │
│  │             │  │             │  │   Accounts  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Policy Engine                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Roles     │  │   Permissions│  │   Resources │ │ │
│  │  │             │  │             │  │             │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   GCP       │  │   External  │  │   Audit     │     │
│  │   Services  │  │   Identity  │  │   Logs      │     │
│  │             │  │   Providers │  │             │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### IAM with Terraform

```hcl
# iam.tf
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

variable "organization_id" {
  description = "GCP Organization ID"
  type        = string
}

# Service Accounts
resource "google_service_account" "app_service" {
  account_id   = "${var.environment}-app-service"
  display_name = "${var.environment} App Service Account"
  description  = "Service account for application services"
  project      = var.project_id
}

resource "google_service_account" "cicd_service" {
  account_id   = "${var.environment}-cicd-service"
  display_name = "${var.environment} CI/CD Service Account"
  description  = "Service account for CI/CD pipelines"
  project      = var.project_id
}

resource "google_service_account" "monitoring_service" {
  account_id   = "${var.environment}-monitoring-service"
  display_name = "${var.environment} Monitoring Service Account"
  description  = "Service account for monitoring services"
  project      = var.project_id
}

# Custom Roles
resource "google_project_iam_custom_role" "app_developer" {
  role_id     = "app_developer"
  title       = "Application Developer"
  description = "Custom role for application developers"
  project     = var.project_id

  permissions = [
    "compute.instances.create",
    "compute.instances.delete",
    "compute.instances.get",
    "compute.instances.list",
    "compute.instances.start",
    "compute.instances.stop",
    "compute.instances.update",
    "storage.buckets.create",
    "storage.buckets.delete",
    "storage.buckets.get",
    "storage.buckets.list",
    "storage.buckets.update",
    "storage.objects.create",
    "storage.objects.delete",
    "storage.objects.get",
    "storage.objects.list",
    "storage.objects.update",
    "sql.instances.create",
    "sql.instances.delete",
    "sql.instances.get",
    "sql.instances.list",
    "sql.instances.update",
    "sql.databases.create",
    "sql.databases.delete",
    "sql.databases.get",
    "sql.databases.list",
    "sql.databases.update",
    "sql.users.create",
    "sql.users.delete",
    "sql.users.get",
    "sql.users.list",
    "sql.users.update",
  ]
}

resource "google_project_iam_custom_role" "data_analyst" {
  role_id     = "data_analyst"
  title       = "Data Analyst"
  description = "Custom role for data analysts"
  project     = var.project_id

  permissions = [
    "bigquery.datasets.create",
    "bigquery.datasets.get",
    "bigquery.datasets.list",
    "bigquery.datasets.update",
    "bigquery.tables.create",
    "bigquery.tables.delete",
    "bigquery.tables.get",
    "bigquery.tables.list",
    "bigquery.tables.update",
    "bigquery.jobs.create",
    "bigquery.jobs.get",
    "bigquery.jobs.list",
    "bigquery.routines.create",
    "bigquery.routines.delete",
    "bigquery.routines.get",
    "bigquery.routines.list",
    "bigquery.routines.update",
    "storage.buckets.get",
    "storage.buckets.list",
    "storage.objects.get",
    "storage.objects.list",
  ]
}

resource "google_project_iam_custom_role" "security_auditor" {
  role_id     = "security_auditor"
  title       = "Security Auditor"
  description = "Custom role for security auditors"
  project     = var.project_id

  permissions = [
    "iam.roles.get",
    "iam.roles.list",
    "iam.serviceAccounts.get",
    "iam.serviceAccounts.list",
    "iam.serviceAccountKeys.list",
    "resourcemanager.projects.get",
    "resourcemanager.projects.list",
    "cloudasset.assets.searchAllResources",
    "cloudasset.assets.searchAllIamPolicies",
    "logging.logs.list",
    "logging.logEntries.list",
    "monitoring.timeSeries.list",
    "monitoring.metricDescriptors.list",
    "securitycenter.assets.list",
    "securitycenter.findings.list",
    "securitycenter.sources.list",
  ]
}

# IAM Bindings for Service Accounts
resource "google_project_iam_binding" "app_service_storage" {
  project = var.project_id
  role    = "roles/storage.admin"

  members = [
    "serviceAccount:${google_service_account.app_service.email}",
  ]
}

resource "google_project_iam_binding" "app_service_sql" {
  project = var.project_id
  role    = "roles/cloudsql.client"

  members = [
    "serviceAccount:${google_service_account.app_service.email}",
  ]
}

resource "google_project_iam_binding" "app_service_secret_manager" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"

  members = [
    "serviceAccount:${google_service_account.app_service.email}",
  ]
}

resource "google_project_iam_binding" "cicd_service_compute" {
  project = var.project_id
  role    = "roles/compute.instanceAdmin"

  members = [
    "serviceAccount:${google_service_account.cicd_service.email}",
  ]
}

resource "google_project_iam_binding" "cicd_service_container" {
  project = var.project_id
  role    = "roles/container.developer"

  members = [
    "serviceAccount:${google_service_account.cicd_service.email}",
  ]
}

resource "google_project_iam_binding" "monitoring_service_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"

  members = [
    "serviceAccount:${google_service_account.monitoring_service.email}",
  ]
}

resource "google_project_iam_binding" "monitoring_service_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"

  members = [
    "serviceAccount:${google_service_account.monitoring_service.email}",
  ]
}

# IAM Bindings for Users and Groups
resource "google_project_iam_binding" "developers" {
  project = var.project_id
  role    = google_project_iam_custom_role.app_developer.id

  members = [
    "group:developers@company.com",
  ]
}

resource "google_project_iam_binding" "data_analysts" {
  project = var.project_id
  role    = google_project_iam_custom_role.data_analyst.id

  members = [
    "group:data-analysts@company.com",
  ]
}

resource "google_project_iam_binding" "security_auditors" {
  project = var.project_id
  role    = google_project_iam_custom_role.security_auditor.id

  members = [
    "group:security-auditors@company.com",
  ]
}

resource "google_project_iam_binding" "project_viewers" {
  project = var.project_id
  role    = "roles/viewer"

  members = [
    "group:project-viewers@company.com",
  ]
}

# Organization Policies
resource "google_organization_policy" "restrict_compute_engine" {
  org_id     = var.organization_id
  constraint = "compute.vmExternalIpAccess"

  list_policy {
    allow {
      all = false
    }
  }
}

resource "google_organization_policy" "restrict_cloud_sql" {
  org_id     = var.organization_id
  constraint = "sql.restrictPublicIp"

  boolean_policy {
    enforced = true
  }
}

resource "google_organization_policy" "require_os_login" {
  org_id     = var.organization_id
  constraint = "compute.requireOsLogin"

  boolean_policy {
    enforced = true
  }
}

resource "google_organization_policy" "restrict_service_account_creation" {
  org_id     = var.organization_id
  constraint = "iam.allowedPolicyMemberDomains"

  list_policy {
    allow {
      values = ["company.com"]
    }
  }
}

# Service Account Keys (for external applications)
resource "google_service_account_key" "app_service_key" {
  service_account_id = google_service_account.app_service.name
  public_key_type    = "TYPE_X509_PEM_FILE"
}

# Workload Identity (for GKE)
resource "google_service_account_iam_binding" "workload_identity" {
  service_account_id = google_service_account.app_service.name
  role               = "roles/iam.workloadIdentityUser"

  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[default/app-service]",
  ]
}

# IAM Policy for Project
resource "google_project_iam_policy" "project_policy" {
  project     = var.project_id
  policy_data = data.google_iam_policy.project_policy.policy_data
}

data "google_iam_policy" "project_policy" {
  binding {
    role = "roles/editor"
    members = [
      "serviceAccount:${google_service_account.app_service.email}",
    ]
  }

  binding {
    role = "roles/container.developer"
    members = [
      "serviceAccount:${google_service_account.cicd_service.email}",
    ]
  }

  binding {
    role = "roles/monitoring.metricWriter"
    members = [
      "serviceAccount:${google_service_account.monitoring_service.email}",
    ]
  }
}

# Audit Logging
resource "google_project_iam_audit_config" "audit_config" {
  project = var.project_id
  service = "allServices"

  audit_log_config {
    log_type = "ADMIN_READ"
  }

  audit_log_config {
    log_type = "DATA_READ"
  }

  audit_log_config {
    log_type = "DATA_WRITE"
  }
}

# Conditional IAM Binding
resource "google_project_iam_member" "conditional_access" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "user:conditional-user@company.com"

  condition {
    title       = "Access only during business hours"
    description = "Access granted only during business hours"
    expression  = "request.time.getHours() >= 9 && request.time.getHours() <= 17"
  }
}

# Outputs
output "app_service_account_email" {
  description = "App service account email"
  value       = google_service_account.app_service.email
}

output "cicd_service_account_email" {
  description = "CI/CD service account email"
  value       = google_service_account.cicd_service.email
}

output "monitoring_service_account_email" {
  description = "Monitoring service account email"
  value       = google_service_account.monitoring_service.email
}

output "app_service_account_key" {
  description = "App service account key"
  value       = google_service_account_key.app_service_key.private_key
  sensitive   = true
}
```

### Go Application with IAM Integration

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

    "cloud.google.com/go/iam/admin/apiv1"
    "cloud.google.com/go/iam/admin/apiv1/adminpb"
    "github.com/gin-gonic/gin"
    "google.golang.org/api/cloudresourcemanager/v1"
    "google.golang.org/api/option"
)

type IAMService struct {
    adminClient *admin.IamClient
    crmService  *cloudresourcemanager.Service
    projectID   string
}

func NewIAMService(projectID string) (*IAMService, error) {
    ctx := context.Background()

    // Initialize IAM Admin client
    adminClient, err := admin.NewIamClient(ctx)
    if err != nil {
        return nil, fmt.Errorf("failed to create IAM admin client: %w", err)
    }

    // Initialize Cloud Resource Manager service
    crmService, err := cloudresourcemanager.NewService(ctx)
    if err != nil {
        return nil, fmt.Errorf("failed to create CRM service: %w", err)
    }

    return &IAMService{
        adminClient: adminClient,
        crmService:  crmService,
        projectID:   projectID,
    }, nil
}

func (s *IAMService) ListServiceAccounts(ctx context.Context) ([]*adminpb.ServiceAccount, error) {
    req := &adminpb.ListServiceAccountsRequest{
        Name: fmt.Sprintf("projects/%s", s.projectID),
    }

    resp, err := s.adminClient.ListServiceAccounts(ctx, req)
    if err != nil {
        return nil, fmt.Errorf("failed to list service accounts: %w", err)
    }

    return resp.Accounts, nil
}

func (s *IAMService) GetServiceAccount(ctx context.Context, email string) (*adminpb.ServiceAccount, error) {
    req := &adminpb.GetServiceAccountRequest{
        Name: fmt.Sprintf("projects/%s/serviceAccounts/%s", s.projectID, email),
    }

    resp, err := s.adminClient.GetServiceAccount(ctx, req)
    if err != nil {
        return nil, fmt.Errorf("failed to get service account: %w", err)
    }

    return resp, nil
}

func (s *IAMService) CreateServiceAccount(ctx context.Context, accountID, displayName, description string) (*adminpb.ServiceAccount, error) {
    req := &adminpb.CreateServiceAccountRequest{
        Name:      fmt.Sprintf("projects/%s", s.projectID),
        AccountId: accountID,
        ServiceAccount: &adminpb.ServiceAccount{
            DisplayName: displayName,
            Description: description,
        },
    }

    resp, err := s.adminClient.CreateServiceAccount(ctx, req)
    if err != nil {
        return nil, fmt.Errorf("failed to create service account: %w", err)
    }

    return resp, nil
}

func (s *IAMService) DeleteServiceAccount(ctx context.Context, email string) error {
    req := &adminpb.DeleteServiceAccountRequest{
        Name: fmt.Sprintf("projects/%s/serviceAccounts/%s", s.projectID, email),
    }

    err := s.adminClient.DeleteServiceAccount(ctx, req)
    if err != nil {
        return fmt.Errorf("failed to delete service account: %w", err)
    }

    return nil
}

func (s *IAMService) ListRoles(ctx context.Context) ([]*adminpb.Role, error) {
    req := &adminpb.ListRolesRequest{
        Parent: fmt.Sprintf("projects/%s", s.projectID),
    }

    resp, err := s.adminClient.ListRoles(ctx, req)
    if err != nil {
        return nil, fmt.Errorf("failed to list roles: %w", err)
    }

    return resp.Roles, nil
}

func (s *IAMService) GetProjectIAMPolicy(ctx context.Context) (*cloudresourcemanager.Policy, error) {
    req := &cloudresourcemanager.GetIamPolicyRequest{}

    policy, err := s.crmService.Projects.GetIamPolicy(s.projectID, req).Context(ctx).Do()
    if err != nil {
        return nil, fmt.Errorf("failed to get IAM policy: %w", err)
    }

    return policy, nil
}

func (s *IAMService) SetProjectIAMPolicy(ctx context.Context, policy *cloudresourcemanager.Policy) (*cloudresourcemanager.Policy, error) {
    req := &cloudresourcemanager.SetIamPolicyRequest{
        Policy: policy,
    }

    updatedPolicy, err := s.crmService.Projects.SetIamPolicy(s.projectID, req).Context(ctx).Do()
    if err != nil {
        return nil, fmt.Errorf("failed to set IAM policy: %w", err)
    }

    return updatedPolicy, nil
}

func (s *IAMService) AddIAMBinding(ctx context.Context, role string, members []string) error {
    // Get current policy
    policy, err := s.GetProjectIAMPolicy(ctx)
    if err != nil {
        return fmt.Errorf("failed to get current policy: %w", err)
    }

    // Find existing binding for the role
    var binding *cloudresourcemanager.Binding
    for _, b := range policy.Bindings {
        if b.Role == role {
            binding = b
            break
        }
    }

    if binding == nil {
        // Create new binding
        binding = &cloudresourcemanager.Binding{
            Role:    role,
            Members: members,
        }
        policy.Bindings = append(policy.Bindings, binding)
    } else {
        // Add members to existing binding
        for _, member := range members {
            found := false
            for _, existingMember := range binding.Members {
                if existingMember == member {
                    found = true
                    break
                }
            }
            if !found {
                binding.Members = append(binding.Members, member)
            }
        }
    }

    // Set updated policy
    _, err = s.SetProjectIAMPolicy(ctx, policy)
    if err != nil {
        return fmt.Errorf("failed to set updated policy: %w", err)
    }

    return nil
}

func (s *IAMService) RemoveIAMBinding(ctx context.Context, role string, members []string) error {
    // Get current policy
    policy, err := s.GetProjectIAMPolicy(ctx)
    if err != nil {
        return fmt.Errorf("failed to get current policy: %w", err)
    }

    // Find binding for the role
    for i, binding := range policy.Bindings {
        if binding.Role == role {
            // Remove specified members
            var updatedMembers []string
            for _, existingMember := range binding.Members {
                shouldRemove := false
                for _, memberToRemove := range members {
                    if existingMember == memberToRemove {
                        shouldRemove = true
                        break
                    }
                }
                if !shouldRemove {
                    updatedMembers = append(updatedMembers, existingMember)
                }
            }

            if len(updatedMembers) == 0 {
                // Remove binding if no members left
                policy.Bindings = append(policy.Bindings[:i], policy.Bindings[i+1:]...)
            } else {
                binding.Members = updatedMembers
            }
            break
        }
    }

    // Set updated policy
    _, err = s.SetProjectIAMPolicy(ctx, policy)
    if err != nil {
        return fmt.Errorf("failed to set updated policy: %w", err)
    }

    return nil
}

func (s *IAMService) TestIAMPermissions(ctx context.Context, permissions []string) ([]string, error) {
    req := &cloudresourcemanager.TestIamPermissionsRequest{
        Permissions: permissions,
    }

    resp, err := s.crmService.Projects.TestIamPermissions(s.projectID, req).Context(ctx).Do()
    if err != nil {
        return nil, fmt.Errorf("failed to test IAM permissions: %w", err)
    }

    return resp.Permissions, nil
}

func (s *IAMService) Close() error {
    return s.adminClient.Close()
}

// HTTP handlers
func setupRoutes(iamService *IAMService) *gin.Engine {
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
        // Service Account routes
        api.GET("/service-accounts", func(c *gin.Context) {
            accounts, err := iamService.ListServiceAccounts(c.Request.Context())
            if err != nil {
                log.Printf("Error listing service accounts: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to list service accounts"})
                return
            }

            var accountList []gin.H
            for _, account := range accounts {
                accountList = append(accountList, gin.H{
                    "name":         account.Name,
                    "email":        account.Email,
                    "display_name": account.DisplayName,
                    "description":  account.Description,
                    "disabled":     account.Disabled,
                })
            }

            c.JSON(http.StatusOK, gin.H{
                "service_accounts": accountList,
                "count":           len(accountList),
            })
        })

        api.GET("/service-accounts/:email", func(c *gin.Context) {
            email := c.Param("email")

            account, err := iamService.GetServiceAccount(c.Request.Context(), email)
            if err != nil {
                log.Printf("Error getting service account: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get service account"})
                return
            }

            c.JSON(http.StatusOK, gin.H{
                "name":         account.Name,
                "email":        account.Email,
                "display_name": account.DisplayName,
                "description":  account.Description,
                "disabled":     account.Disabled,
            })
        })

        api.POST("/service-accounts", func(c *gin.Context) {
            var req struct {
                AccountID   string `json:"account_id" binding:"required"`
                DisplayName string `json:"display_name" binding:"required"`
                Description string `json:"description"`
            }

            if err := c.ShouldBindJSON(&req); err != nil {
                c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
                return
            }

            account, err := iamService.CreateServiceAccount(c.Request.Context(), req.AccountID, req.DisplayName, req.Description)
            if err != nil {
                log.Printf("Error creating service account: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create service account"})
                return
            }

            c.JSON(http.StatusCreated, gin.H{
                "name":         account.Name,
                "email":        account.Email,
                "display_name": account.DisplayName,
                "description":  account.Description,
            })
        })

        api.DELETE("/service-accounts/:email", func(c *gin.Context) {
            email := c.Param("email")

            err := iamService.DeleteServiceAccount(c.Request.Context(), email)
            if err != nil {
                log.Printf("Error deleting service account: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to delete service account"})
                return
            }

            c.JSON(http.StatusNoContent, nil)
        })

        // Role routes
        api.GET("/roles", func(c *gin.Context) {
            roles, err := iamService.ListRoles(c.Request.Context())
            if err != nil {
                log.Printf("Error listing roles: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to list roles"})
                return
            }

            var roleList []gin.H
            for _, role := range roles {
                roleList = append(roleList, gin.H{
                    "name":        role.Name,
                    "title":       role.Title,
                    "description": role.Description,
                    "stage":       role.Stage.String(),
                })
            }

            c.JSON(http.StatusOK, gin.H{
                "roles": roleList,
                "count": len(roleList),
            })
        })

        // IAM Policy routes
        api.GET("/iam-policy", func(c *gin.Context) {
            policy, err := iamService.GetProjectIAMPolicy(c.Request.Context())
            if err != nil {
                log.Printf("Error getting IAM policy: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get IAM policy"})
                return
            }

            c.JSON(http.StatusOK, gin.H{
                "bindings": policy.Bindings,
                "etag":     policy.Etag,
            })
        })

        api.POST("/iam-bindings", func(c *gin.Context) {
            var req struct {
                Role    string   `json:"role" binding:"required"`
                Members []string `json:"members" binding:"required"`
            }

            if err := c.ShouldBindJSON(&req); err != nil {
                c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
                return
            }

            err := iamService.AddIAMBinding(c.Request.Context(), req.Role, req.Members)
            if err != nil {
                log.Printf("Error adding IAM binding: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to add IAM binding"})
                return
            }

            c.JSON(http.StatusCreated, gin.H{
                "message": "IAM binding added successfully",
                "role":    req.Role,
                "members": req.Members,
            })
        })

        api.DELETE("/iam-bindings", func(c *gin.Context) {
            var req struct {
                Role    string   `json:"role" binding:"required"`
                Members []string `json:"members" binding:"required"`
            }

            if err := c.ShouldBindJSON(&req); err != nil {
                c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
                return
            }

            err := iamService.RemoveIAMBinding(c.Request.Context(), req.Role, req.Members)
            if err != nil {
                log.Printf("Error removing IAM binding: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to remove IAM binding"})
                return
            }

            c.JSON(http.StatusOK, gin.H{
                "message": "IAM binding removed successfully",
                "role":    req.Role,
                "members": req.Members,
            })
        })

        // Permission testing
        api.POST("/test-permissions", func(c *gin.Context) {
            var req struct {
                Permissions []string `json:"permissions" binding:"required"`
            }

            if err := c.ShouldBindJSON(&req); err != nil {
                c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
                return
            }

            permissions, err := iamService.TestIAMPermissions(c.Request.Context(), req.Permissions)
            if err != nil {
                log.Printf("Error testing IAM permissions: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to test IAM permissions"})
                return
            }

            c.JSON(http.StatusOK, gin.H{
                "requested_permissions": req.Permissions,
                "granted_permissions":   permissions,
                "denied_permissions":    getDeniedPermissions(req.Permissions, permissions),
            })
        })
    }

    return r
}

func getDeniedPermissions(requested, granted []string) []string {
    grantedMap := make(map[string]bool)
    for _, perm := range granted {
        grantedMap[perm] = true
    }

    var denied []string
    for _, perm := range requested {
        if !grantedMap[perm] {
            denied = append(denied, perm)
        }
    }

    return denied
}

func main() {
    // Get configuration from environment
    projectID := os.Getenv("GCP_PROJECT_ID")
    if projectID == "" {
        log.Fatal("GCP_PROJECT_ID environment variable is required")
    }

    // Initialize IAM service
    iamService, err := NewIAMService(projectID)
    if err != nil {
        log.Fatalf("Failed to initialize IAM service: %v", err)
    }
    defer iamService.Close()

    // Setup routes
    r := setupRoutes(iamService)

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

### 1. Principle of Least Privilege
```hcl
# Grant minimum required permissions
resource "google_project_iam_binding" "minimal_access" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  members = ["user:readonly-user@company.com"]
}
```

### 2. Service Account Management
```hcl
# Use service accounts for applications
resource "google_service_account" "app_service" {
  account_id   = "app-service"
  display_name = "Application Service Account"
  description  = "Service account for application services"
}
```

### 3. Organization Policies
```hcl
# Enforce security policies
resource "google_organization_policy" "restrict_public_ip" {
  org_id     = var.organization_id
  constraint = "compute.vmExternalIpAccess"

  list_policy {
    allow {
      all = false
    }
  }
}
```

## 🏢 Industry Insights

### IAM Usage Patterns
- **Role-Based Access Control**: Assign permissions to roles, not users
- **Service Accounts**: Identity for applications and services
- **Organization Policies**: Centralized policy enforcement
- **Audit Logging**: Comprehensive access monitoring

### Enterprise IAM Strategy
- **Centralized Management**: Single source of truth for access control
- **Automated Provisioning**: Automated user lifecycle management
- **Compliance**: Audit trails and policy enforcement
- **Security**: Multi-factor authentication and conditional access

## 🎯 Interview Questions

### Basic Level
1. **What is GCP IAM?**
   - Identity and Access Management service
   - Fine-grained access control
   - Role-based permissions

2. **What are the main IAM components?**
   - Users and groups
   - Service accounts
   - Roles and permissions
   - Policies

3. **What are service accounts?**
   - Identity for applications
   - Non-human accounts
   - Used for service-to-service authentication

### Intermediate Level
4. **How do you implement least privilege access?**
   - Grant minimum required permissions
   - Use custom roles
   - Regular access reviews
   - Principle of least privilege

5. **How do you manage service accounts?**
   - Create service accounts for applications
   - Use workload identity
   - Rotate service account keys
   - Monitor service account usage

6. **How do you implement organization policies?**
   - Define constraints
   - Enforce policies across projects
   - Use boolean and list policies
   - Monitor policy violations

### Advanced Level
7. **How do you implement IAM at scale?**
   - Automated provisioning
   - Role-based access control
   - Centralized policy management
   - Audit and compliance

8. **How do you handle IAM security?**
   - Multi-factor authentication
   - Conditional access
   - Audit logging
   - Policy enforcement

9. **How do you implement IAM governance?**
   - Access reviews
   - Policy compliance
   - Audit trails
   - Risk management
