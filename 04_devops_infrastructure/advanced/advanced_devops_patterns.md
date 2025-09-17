# Advanced DevOps Patterns

Advanced DevOps patterns and practices for senior engineering roles.

## 🎯 GitOps and Infrastructure as Code

### GitOps Principles
- **Declarative**: Infrastructure defined as code
- **Version Controlled**: All changes in Git
- **Automated**: Continuous deployment
- **Observable**: Full audit trail

### Infrastructure as Code
```yaml
# Terraform example
resource "aws_eks_cluster" "main" {
  name     = var.cluster_name
  role_arn = aws_iam_role.cluster.arn
  version  = var.kubernetes_version

  vpc_config {
    subnet_ids = var.subnet_ids
  }

  depends_on = [
    aws_iam_role_policy_attachment.cluster_AmazonEKSClusterPolicy,
  ]
}
```

## 🔄 CI/CD Advanced Patterns

### Multi-Stage Pipelines
```yaml
# GitLab CI example
stages:
  - build
  - test
  - security
  - deploy

build:
  stage: build
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

test:
  stage: test
  script:
    - docker run --rm $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA npm test

security:
  stage: security
  script:
    - docker run --rm -v $(pwd):/app $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA npm audit
```

### Blue-Green Deployment
```bash
#!/bin/bash
# Blue-Green deployment script

# Deploy to green environment
kubectl apply -f green-deployment.yaml

# Wait for green to be ready
kubectl wait --for=condition=available deployment/green-app --timeout=300s

# Switch traffic to green
kubectl patch service app-service -p '{"spec":{"selector":{"version":"green"}}}'

# Monitor green for 5 minutes
sleep 300

# If healthy, remove blue
kubectl delete deployment blue-app
```

## 🚀 Advanced Monitoring

### Distributed Tracing
```go
// OpenTelemetry example
package main

import (
    "context"
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/trace"
)

func main() {
    tracer := otel.Tracer("my-service")
    
    ctx, span := tracer.Start(context.Background(), "operation")
    defer span.End()
    
    // Your business logic here
    result := processRequest(ctx)
    
    span.SetAttributes(
        attribute.String("result", result),
    )
}
```

### Custom Metrics
```go
// Prometheus metrics example
import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

var (
    httpRequestsTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total number of HTTP requests",
        },
        []string{"method", "endpoint", "status"},
    )
    
    httpRequestDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "http_request_duration_seconds",
            Help: "Duration of HTTP requests",
        },
        []string{"method", "endpoint"},
    )
)
```

## 🔐 Security Patterns

### Zero Trust Architecture
```yaml
# Istio security policy
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-nothing
  namespace: default
spec:
  {}

---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-frontend
  namespace: default
spec:
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/default/sa/frontend"]
    to:
    - operation:
        methods: ["GET", "POST"]
        paths: ["/api/*"]
```

### Secrets Management
```go
// Vault integration example
package main

import (
    "github.com/hashicorp/vault/api"
)

type VaultClient struct {
    client *api.Client
}

func (v *VaultClient) GetSecret(path string) (map[string]interface{}, error) {
    secret, err := v.client.Logical().Read(path)
    if err != nil {
        return nil, err
    }
    
    return secret.Data, nil
}

func (v *VaultClient) RenewToken() error {
    _, err := v.client.Auth().Token().RenewSelf(0)
    return err
}
```

## 📊 Performance Optimization

### Auto-scaling
```yaml
# HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: app-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Resource Optimization
```yaml
# Resource limits and requests
apiVersion: apps/v1
kind: Deployment
metadata:
  name: optimized-app
spec:
  template:
    spec:
      containers:
      - name: app
        image: app:latest
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
```

## 🔄 Disaster Recovery

### Backup Strategy
```bash
#!/bin/bash
# Database backup script

# Create backup
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > backup_$(date +%Y%m%d_%H%M%S).sql

# Compress backup
gzip backup_$(date +%Y%m%d_%H%M%S).sql

# Upload to S3
aws s3 cp backup_$(date +%Y%m%d_%H%M%S).sql.gz s3://$BACKUP_BUCKET/

# Cleanup old backups (keep 30 days)
find /backups -name "backup_*.sql.gz" -mtime +30 -delete
```

### Multi-Region Deployment
```yaml
# Multi-region Kubernetes deployment
apiVersion: v1
kind: ConfigMap
metadata:
  name: multi-region-config
data:
  primary-region: "us-west-2"
  secondary-region: "us-east-1"
  failover-threshold: "5"
  health-check-interval: "30s"
```

## 🎯 Best Practices

### Infrastructure as Code
1. **Version Control**: All infrastructure in Git
2. **Modularity**: Reusable components
3. **Testing**: Infrastructure testing
4. **Documentation**: Clear documentation
5. **Review**: Code review process

### Monitoring and Observability
1. **Metrics**: Business and technical metrics
2. **Logging**: Structured logging
3. **Tracing**: Distributed tracing
4. **Alerting**: Smart alerting rules
5. **Dashboards**: Real-time visibility

### Security
1. **Least Privilege**: Minimal permissions
2. **Encryption**: Data at rest and in transit
3. **Secrets**: Secure secret management
4. **Audit**: Regular security audits
5. **Compliance**: Regulatory compliance

---

**Last Updated**: December 2024  
**Category**: Advanced DevOps Patterns  
**Complexity**: Senior Level
