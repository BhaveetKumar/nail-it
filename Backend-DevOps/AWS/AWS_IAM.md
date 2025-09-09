# 🔐 AWS IAM: Identity and Access Management

> **Master AWS IAM for secure access control and identity management**

## 📚 Concept

AWS Identity and Access Management (IAM) is a web service that helps you securely control access to AWS resources. You use IAM to control who is authenticated (signed in) and authorized (has permissions) to use resources.

### Key Features

- **Users and Groups**: Identity management
- **Roles**: Temporary access credentials
- **Policies**: Permission management
- **Multi-Factor Authentication**: Enhanced security
- **Access Keys**: Programmatic access
- **Federation**: External identity integration

## 🏗️ IAM Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    IAM Architecture                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Users     │  │   Groups    │  │   Roles     │     │
│  │             │  │             │  │             │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Policy Engine                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Identity  │  │   Resource  │  │   Access    │ │ │
│  │  │   Policies  │  │   Policies  │  │   Policies  │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   AWS       │  │   External  │  │   Cross     │     │
│  │   Services  │  │   Identity  │  │   Account   │     │
│  │             │  │   Providers │  │   Access    │     │
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

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "myapp"
}

# IAM Users
resource "aws_iam_user" "developers" {
  for_each = toset([
    "john.doe",
    "jane.smith",
    "bob.wilson"
  ])

  name = each.key
  path = "/developers/"

  tags = {
    Name        = each.key
    Environment = var.environment
    Project     = var.project_name
    Role        = "developer"
  }
}

resource "aws_iam_user" "admins" {
  for_each = toset([
    "admin.user",
    "security.admin"
  ])

  name = each.key
  path = "/admins/"

  tags = {
    Name        = each.key
    Environment = var.environment
    Project     = var.project_name
    Role        = "admin"
  }
}

# IAM Groups
resource "aws_iam_group" "developers" {
  name = "${var.project_name}-developers"
  path = "/groups/"
}

resource "aws_iam_group" "admins" {
  name = "${var.project_name}-admins"
  path = "/groups/"
}

resource "aws_iam_group" "readonly" {
  name = "${var.project_name}-readonly"
  path = "/groups/"
}

# Group Memberships
resource "aws_iam_user_group_membership" "developers" {
  for_each = aws_iam_user.developers
  user     = each.value.name
  groups   = [aws_iam_group.developers.name]
}

resource "aws_iam_user_group_membership" "admins" {
  for_each = aws_iam_user.admins
  user     = each.value.name
  groups   = [aws_iam_group.admins.name]
}

# IAM Policies
resource "aws_iam_policy" "developer_policy" {
  name        = "${var.project_name}-developer-policy"
  description = "Policy for developers"
  path        = "/policies/"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:Describe*",
          "ec2:Get*",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:PutObject",
          "s3:DeleteObject",
          "rds:Describe*",
          "rds:List*",
          "cloudformation:Describe*",
          "cloudformation:List*",
          "cloudformation:Get*"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-developer-policy"
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_iam_policy" "admin_policy" {
  name        = "${var.project_name}-admin-policy"
  description = "Policy for administrators"
  path        = "/policies/"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "*"
        Resource = "*"
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-admin-policy"
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_iam_policy" "readonly_policy" {
  name        = "${var.project_name}-readonly-policy"
  description = "Policy for read-only access"
  path        = "/policies/"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:Describe*",
          "ec2:Get*",
          "s3:GetObject",
          "s3:ListBucket",
          "rds:Describe*",
          "rds:List*",
          "cloudformation:Describe*",
          "cloudformation:List*",
          "cloudformation:Get*",
          "iam:Get*",
          "iam:List*"
        ]
        Resource = "*"
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-readonly-policy"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Attach policies to groups
resource "aws_iam_group_policy_attachment" "developers" {
  group      = aws_iam_group.developers.name
  policy_arn = aws_iam_policy.developer_policy.arn
}

resource "aws_iam_group_policy_attachment" "admins" {
  group      = aws_iam_group.admins.name
  policy_arn = aws_iam_policy.admin_policy.arn
}

resource "aws_iam_group_policy_attachment" "readonly" {
  group      = aws_iam_group.readonly.name
  policy_arn = aws_iam_policy.readonly_policy.arn
}

# IAM Roles
resource "aws_iam_role" "ec2_role" {
  name = "${var.project_name}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-ec2-role"
    Environment = var.environment
    Project     = var.project_name
  }
}

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

  tags = {
    Name        = "${var.project_name}-lambda-role"
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_iam_role" "cross_account_role" {
  name = "${var.project_name}-cross-account-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::123456789012:root"
        }
        Condition = {
          StringEquals = {
            "sts:ExternalId" = "unique-external-id"
          }
        }
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-cross-account-role"
    Environment = var.environment
    Project     = var.project_name
  }
}

# IAM Policies for Roles
resource "aws_iam_policy" "ec2_policy" {
  name        = "${var.project_name}-ec2-policy"
  description = "Policy for EC2 instances"
  path        = "/policies/"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.project_name}-*",
          "arn:aws:s3:::${var.project_name}-*/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters",
          "ssm:GetParametersByPath"
        ]
        Resource = "arn:aws:ssm:*:*:parameter/${var.project_name}/*"
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-ec2-policy"
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_iam_policy" "lambda_policy" {
  name        = "${var.project_name}-lambda-policy"
  description = "Policy for Lambda functions"
  path        = "/policies/"

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
        Resource = "arn:aws:dynamodb:*:*:table/${var.project_name}-*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.project_name}-*",
          "arn:aws:s3:::${var.project_name}-*/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ses:SendEmail",
          "ses:SendRawEmail"
        ]
        Resource = "*"
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-lambda-policy"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Attach policies to roles
resource "aws_iam_role_policy_attachment" "ec2_role" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.ec2_policy.arn
}

resource "aws_iam_role_policy_attachment" "lambda_role" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = aws_iam_policy.lambda_policy.arn
}

# Attach AWS managed policies
resource "aws_iam_role_policy_attachment" "ec2_ssm" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Instance Profile
resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project_name}-ec2-profile"
  role = aws_iam_role.ec2_role.name

  tags = {
    Name        = "${var.project_name}-ec2-profile"
    Environment = var.environment
    Project     = var.project_name
  }
}

# MFA Policy
resource "aws_iam_policy" "mfa_policy" {
  name        = "${var.project_name}-mfa-policy"
  description = "Policy requiring MFA for sensitive operations"
  path        = "/policies/"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "DenyAllExceptListedIfNoMFA"
        Effect = "Deny"
        NotAction = [
          "iam:CreateVirtualMFADevice",
          "iam:EnableMFADevice",
          "iam:GetUser",
          "iam:ListMFADevices",
          "iam:ListVirtualMFADevices",
          "iam:ResyncMFADevice",
          "sts:GetSessionToken"
        ]
        Resource = "*"
        Condition = {
          BoolIfExists = {
            "aws:MultiFactorAuthPresent" = "false"
          }
        }
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-mfa-policy"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Attach MFA policy to admin group
resource "aws_iam_group_policy_attachment" "admins_mfa" {
  group      = aws_iam_group.admins.name
  policy_arn = aws_iam_policy.mfa_policy.arn
}

# Access Keys (for service accounts)
resource "aws_iam_access_key" "service_account" {
  user = aws_iam_user.developers["john.doe"].name
}

# Outputs
output "ec2_role_arn" {
  description = "EC2 role ARN"
  value       = aws_iam_role.ec2_role.arn
}

output "lambda_role_arn" {
  description = "Lambda role ARN"
  value       = aws_iam_role.lambda_role.arn
}

output "cross_account_role_arn" {
  description = "Cross-account role ARN"
  value       = aws_iam_role.cross_account_role.arn
}

output "instance_profile_name" {
  description = "Instance profile name"
  value       = aws_iam_instance_profile.ec2_profile.name
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

    "github.com/aws/aws-sdk-go/aws"
    "github.com/aws/aws-sdk-go/aws/session"
    "github.com/aws/aws-sdk-go/service/iam"
    "github.com/aws/aws-sdk-go/service/s3"
    "github.com/aws/aws-sdk-go/service/sts"
    "github.com/gin-gonic/gin"
)

type IAMService struct {
    iamClient *iam.IAM
    s3Client  *s3.S3
    stsClient *sts.STS
}

func NewIAMService() (*IAMService, error) {
    sess, err := session.NewSession(&aws.Config{
        Region: aws.String("us-east-1"),
    })
    if err != nil {
        return nil, fmt.Errorf("failed to create session: %w", err)
    }

    return &IAMService{
        iamClient: iam.New(sess),
        s3Client:  s3.New(sess),
        stsClient: sts.New(sess),
    }, nil
}

func (s *IAMService) GetCurrentUser(ctx context.Context) (*sts.GetCallerIdentityOutput, error) {
    result, err := s.stsClient.GetCallerIdentityWithContext(ctx, &sts.GetCallerIdentityInput{})
    if err != nil {
        return nil, fmt.Errorf("failed to get caller identity: %w", err)
    }

    return result, nil
}

func (s *IAMService) ListUsers(ctx context.Context) ([]*iam.User, error) {
    result, err := s.iamClient.ListUsersWithContext(ctx, &iam.ListUsersInput{})
    if err != nil {
        return nil, fmt.Errorf("failed to list users: %w", err)
    }

    return result.Users, nil
}

func (s *IAMService) ListGroups(ctx context.Context) ([]*iam.Group, error) {
    result, err := s.iamClient.ListGroupsWithContext(ctx, &iam.ListGroupsInput{})
    if err != nil {
        return nil, fmt.Errorf("failed to list groups: %w", err)
    }

    return result.Groups, nil
}

func (s *IAMService) ListRoles(ctx context.Context) ([]*iam.Role, error) {
    result, err := s.iamClient.ListRolesWithContext(ctx, &iam.ListRolesInput{})
    if err != nil {
        return nil, fmt.Errorf("failed to list roles: %w", err)
    }

    return result.Roles, nil
}

func (s *IAMService) GetUserPolicies(ctx context.Context, userName string) ([]*iam.AttachedPolicy, error) {
    result, err := s.iamClient.ListAttachedUserPoliciesWithContext(ctx, &iam.ListAttachedUserPoliciesInput{
        UserName: aws.String(userName),
    })
    if err != nil {
        return nil, fmt.Errorf("failed to list user policies: %w", err)
    }

    return result.AttachedPolicies, nil
}

func (s *IAMService) GetGroupPolicies(ctx context.Context, groupName string) ([]*iam.AttachedPolicy, error) {
    result, err := s.iamClient.ListAttachedGroupPoliciesWithContext(ctx, &iam.ListAttachedGroupPoliciesInput{
        GroupName: aws.String(groupName),
    })
    if err != nil {
        return nil, fmt.Errorf("failed to list group policies: %w", err)
    }

    return result.AttachedPolicies, nil
}

func (s *IAMService) GetRolePolicies(ctx context.Context, roleName string) ([]*iam.AttachedPolicy, error) {
    result, err := s.iamClient.ListAttachedRolePoliciesWithContext(ctx, &iam.ListAttachedRolePoliciesInput{
        RoleName: aws.String(roleName),
    })
    if err != nil {
        return nil, fmt.Errorf("failed to list role policies: %w", err)
    }

    return result.AttachedPolicies, nil
}

func (s *IAMService) AssumeRole(ctx context.Context, roleArn, sessionName string) (*sts.AssumeRoleOutput, error) {
    result, err := s.stsClient.AssumeRoleWithContext(ctx, &sts.AssumeRoleInput{
        RoleArn:         aws.String(roleArn),
        RoleSessionName: aws.String(sessionName),
        DurationSeconds: aws.Int64(3600), // 1 hour
    })
    if err != nil {
        return nil, fmt.Errorf("failed to assume role: %w", err)
    }

    return result, nil
}

func (s *IAMService) GetSessionToken(ctx context.Context, durationSeconds int64) (*sts.GetSessionTokenOutput, error) {
    result, err := s.stsClient.GetSessionTokenWithContext(ctx, &sts.GetSessionTokenInput{
        DurationSeconds: aws.Int64(durationSeconds),
    })
    if err != nil {
        return nil, fmt.Errorf("failed to get session token: %w", err)
    }

    return result, nil
}

func (s *IAMService) ListS3Buckets(ctx context.Context) ([]*s3.Bucket, error) {
    result, err := s.s3Client.ListBucketsWithContext(ctx, &s3.ListBucketsInput{})
    if err != nil {
        return nil, fmt.Errorf("failed to list S3 buckets: %w", err)
    }

    return result.Buckets, nil
}

// HTTP handlers
func setupRoutes(iamService *IAMService) *gin.Engine {
    r := gin.Default()

    // Middleware for authentication
    r.Use(func(c *gin.Context) {
        // In a real application, you would validate JWT tokens or session cookies
        // For this example, we'll just add a simple header check
        if c.GetHeader("Authorization") == "" {
            c.JSON(http.StatusUnauthorized, gin.H{"error": "Authorization header required"})
            c.Abort()
            return
        }
        c.Next()
    })

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
        // Get current user information
        api.GET("/me", func(c *gin.Context) {
            user, err := iamService.GetCurrentUser(c.Request.Context())
            if err != nil {
                log.Printf("Error getting current user: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get user information"})
                return
            }

            c.JSON(http.StatusOK, gin.H{
                "user_id":    user.UserId,
                "account":    user.Account,
                "arn":        user.Arn,
            })
        })

        // List users
        api.GET("/users", func(c *gin.Context) {
            users, err := iamService.ListUsers(c.Request.Context())
            if err != nil {
                log.Printf("Error listing users: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to list users"})
                return
            }

            var userList []gin.H
            for _, user := range users {
                userList = append(userList, gin.H{
                    "user_name": user.UserName,
                    "user_id":   user.UserId,
                    "arn":       user.Arn,
                    "path":      user.Path,
                    "created":   user.CreateDate,
                })
            }

            c.JSON(http.StatusOK, gin.H{
                "users": userList,
                "count": len(userList),
            })
        })

        // List groups
        api.GET("/groups", func(c *gin.Context) {
            groups, err := iamService.ListGroups(c.Request.Context())
            if err != nil {
                log.Printf("Error listing groups: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to list groups"})
                return
            }

            var groupList []gin.H
            for _, group := range groups {
                groupList = append(groupList, gin.H{
                    "group_name": group.GroupName,
                    "group_id":   group.GroupId,
                    "arn":        group.Arn,
                    "path":       group.Path,
                    "created":    group.CreateDate,
                })
            }

            c.JSON(http.StatusOK, gin.H{
                "groups": groupList,
                "count":  len(groupList),
            })
        })

        // List roles
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
                    "role_name": role.RoleName,
                    "role_id":   role.RoleId,
                    "arn":       role.Arn,
                    "path":      role.Path,
                    "created":   role.CreateDate,
                })
            }

            c.JSON(http.StatusOK, gin.H{
                "roles": roleList,
                "count": len(roleList),
            })
        })

        // Get user policies
        api.GET("/users/:username/policies", func(c *gin.Context) {
            username := c.Param("username")

            policies, err := iamService.GetUserPolicies(c.Request.Context(), username)
            if err != nil {
                log.Printf("Error getting user policies: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get user policies"})
                return
            }

            var policyList []gin.H
            for _, policy := range policies {
                policyList = append(policyList, gin.H{
                    "policy_name": policy.PolicyName,
                    "policy_arn":  policy.PolicyArn,
                })
            }

            c.JSON(http.StatusOK, gin.H{
                "username": username,
                "policies": policyList,
                "count":    len(policyList),
            })
        })

        // Get group policies
        api.GET("/groups/:groupname/policies", func(c *gin.Context) {
            groupname := c.Param("groupname")

            policies, err := iamService.GetGroupPolicies(c.Request.Context(), groupname)
            if err != nil {
                log.Printf("Error getting group policies: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get group policies"})
                return
            }

            var policyList []gin.H
            for _, policy := range policies {
                policyList = append(policyList, gin.H{
                    "policy_name": policy.PolicyName,
                    "policy_arn":  policy.PolicyArn,
                })
            }

            c.JSON(http.StatusOK, gin.H{
                "group_name": groupname,
                "policies":   policyList,
                "count":      len(policyList),
            })
        })

        // Get role policies
        api.GET("/roles/:rolename/policies", func(c *gin.Context) {
            rolename := c.Param("rolename")

            policies, err := iamService.GetRolePolicies(c.Request.Context(), rolename)
            if err != nil {
                log.Printf("Error getting role policies: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get role policies"})
                return
            }

            var policyList []gin.H
            for _, policy := range policies {
                policyList = append(policyList, gin.H{
                    "policy_name": policy.PolicyName,
                    "policy_arn":  policy.PolicyArn,
                })
            }

            c.JSON(http.StatusOK, gin.H{
                "role_name": rolename,
                "policies":  policyList,
                "count":     len(policyList),
            })
        })

        // Assume role
        api.POST("/assume-role", func(c *gin.Context) {
            var req struct {
                RoleArn      string `json:"role_arn" binding:"required"`
                SessionName  string `json:"session_name" binding:"required"`
            }

            if err := c.ShouldBindJSON(&req); err != nil {
                c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
                return
            }

            result, err := iamService.AssumeRole(c.Request.Context(), req.RoleArn, req.SessionName)
            if err != nil {
                log.Printf("Error assuming role: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to assume role"})
                return
            }

            c.JSON(http.StatusOK, gin.H{
                "credentials": gin.H{
                    "access_key_id":     result.Credentials.AccessKeyId,
                    "secret_access_key": result.Credentials.SecretAccessKey,
                    "session_token":     result.Credentials.SessionToken,
                    "expiration":        result.Credentials.Expiration,
                },
                "assumed_role_user": gin.H{
                    "assumed_role_id": result.AssumedRoleUser.AssumedRoleId,
                    "arn":             result.AssumedRoleUser.Arn,
                },
            })
        })

        // Get session token
        api.POST("/session-token", func(c *gin.Context) {
            var req struct {
                DurationSeconds int64 `json:"duration_seconds"`
            }

            if err := c.ShouldBindJSON(&req); err != nil {
                c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
                return
            }

            if req.DurationSeconds == 0 {
                req.DurationSeconds = 3600 // Default to 1 hour
            }

            result, err := iamService.GetSessionToken(c.Request.Context(), req.DurationSeconds)
            if err != nil {
                log.Printf("Error getting session token: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get session token"})
                return
            }

            c.JSON(http.StatusOK, gin.H{
                "credentials": gin.H{
                    "access_key_id":     result.Credentials.AccessKeyId,
                    "secret_access_key": result.Credentials.SecretAccessKey,
                    "session_token":     result.Credentials.SessionToken,
                    "expiration":        result.Credentials.Expiration,
                },
            })
        })

        // List S3 buckets (to test permissions)
        api.GET("/s3/buckets", func(c *gin.Context) {
            buckets, err := iamService.ListS3Buckets(c.Request.Context())
            if err != nil {
                log.Printf("Error listing S3 buckets: %v", err)
                c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to list S3 buckets"})
                return
            }

            var bucketList []gin.H
            for _, bucket := range buckets {
                bucketList = append(bucketList, gin.H{
                    "name":         bucket.Name,
                    "creation_date": bucket.CreationDate,
                })
            }

            c.JSON(http.StatusOK, gin.H{
                "buckets": bucketList,
                "count":   len(bucketList),
            })
        })
    }

    return r
}

func main() {
    // Initialize IAM service
    iamService, err := NewIAMService()
    if err != nil {
        log.Fatalf("Failed to initialize IAM service: %v", err)
    }

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

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::my-bucket/*"
    }
  ]
}
```

### 2. MFA Enforcement

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Deny",
      "Action": "*",
      "Resource": "*",
      "Condition": {
        "BoolIfExists": {
          "aws:MultiFactorAuthPresent": "false"
        }
      }
    }
  ]
}
```

### 3. Cross-Account Access

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::123456789012:root"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": {
          "sts:ExternalId": "unique-external-id"
        }
      }
    }
  ]
}
```

## 🏢 Industry Insights

### IAM Usage Patterns

- **Role-Based Access Control**: Assign permissions to roles, not users
- **Temporary Credentials**: Use STS for temporary access
- **Cross-Account Access**: Secure access between AWS accounts
- **Federation**: Integrate with external identity providers

### Enterprise IAM Strategy

- **Centralized Identity Management**: Single source of truth
- **Automated Provisioning**: Automated user lifecycle management
- **Audit and Compliance**: Comprehensive logging and monitoring
- **Security Policies**: Enforce security best practices

## 🎯 Interview Questions

### Basic Level

1. **What is AWS IAM?**

   - Identity and Access Management service
   - User and permission management
   - Security for AWS resources

2. **What are IAM users, groups, and roles?**

   - Users: Individual identities
   - Groups: Collections of users
   - Roles: Temporary access credentials

3. **What are IAM policies?**
   - JSON documents that define permissions
   - Identity-based and resource-based policies
   - Allow or deny access to resources

### Intermediate Level

4. **How do you implement least privilege access?**

   - Grant minimum required permissions
   - Regular access reviews
   - Principle of least privilege

5. **How do you handle cross-account access?**

   - Cross-account roles
   - External ID for additional security
   - Trust relationships

6. **How do you implement MFA?**
   - Virtual MFA devices
   - Hardware MFA devices
   - MFA enforcement policies

### Advanced Level

7. **How do you implement identity federation?**

   - SAML 2.0 federation
   - OpenID Connect
   - Custom identity brokers

8. **How do you handle IAM at scale?**

   - Automated provisioning
   - Role-based access control
   - Centralized identity management

9. **How do you implement IAM security best practices?**
   - Regular access reviews
   - MFA enforcement
   - Audit logging
   - Policy optimization
