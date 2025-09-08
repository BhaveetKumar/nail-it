# 🚀 ArgoCD: GitOps Deployment and Continuous Delivery

> **Master ArgoCD for GitOps-based deployment and continuous delivery**

## 📚 Concept

ArgoCD is a declarative, GitOps continuous delivery tool for Kubernetes. It follows the GitOps pattern where the desired state of your application is defined in Git, and ArgoCD ensures the cluster matches that state.

### Key Features
- **GitOps**: Git as the single source of truth
- **Declarative**: Desired state management
- **Multi-Environment**: Support for multiple environments
- **Sync Policies**: Automated and manual sync options
- **Health Monitoring**: Application health checks
- **Rollback**: Easy rollback to previous versions

## 🏗️ ArgoCD Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Git Repository                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   App       │  │   App       │  │   App       │     │
│  │   Configs   │  │   Configs   │  │   Configs   │     │
│  │   (Staging) │  │   (Prod)    │  │   (Dev)     │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              ArgoCD Server                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   API       │  │   Web UI    │  │   Sync      │ │ │
│  │  │   Server    │  │   Dashboard │  │   Engine    │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Staging   │  │ Production  │  │ Development │     │
│  │   Cluster   │  │   Cluster   │  │   Cluster   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### ArgoCD Application Configuration

```yaml
# argocd-app.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: my-app
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/your-repo.git
    targetRevision: HEAD
    path: k8s/overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
      - PruneLast=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
  revisionHistoryLimit: 10
```

### Kustomize Configuration

```yaml
# k8s/base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - deployment.yaml
  - service.yaml
  - configmap.yaml
  - secret.yaml

commonLabels:
  app: my-app
  version: v1.0.0

images:
  - name: my-app
    newTag: latest
```

```yaml
# k8s/overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../../base

namespace: production

replicas:
  - name: my-app
    count: 3

patchesStrategicMerge:
  - production-patch.yaml

configMapGenerator:
  - name: app-config
    literals:
      - ENV=production
      - LOG_LEVEL=info
```

### Helm Chart Configuration

```yaml
# argocd-helm-app.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: my-app-helm
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/your-repo.git
    targetRevision: HEAD
    path: charts/my-app
    helm:
      valueFiles:
        - values-production.yaml
      parameters:
        - name: image.tag
          value: "v1.0.0"
        - name: replicaCount
          value: "3"
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

### Application Set Configuration

```yaml
# argocd-appset.yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: my-app-set
  namespace: argocd
spec:
  generators:
  - clusters:
      selector:
        matchLabels:
          environment: production
  - clusters:
      selector:
        matchLabels:
          environment: staging
  template:
    metadata:
      name: '{{name}}-{{values.environment}}'
    spec:
      project: default
      source:
        repoURL: https://github.com/your-org/your-repo.git
        targetRevision: HEAD
        path: k8s/overlays/{{values.environment}}
      destination:
        server: '{{server}}'
        namespace: '{{values.environment}}'
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
        syncOptions:
          - CreateNamespace=true
```

### ArgoCD Installation

```yaml
# argocd-install.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: argocd
---
apiVersion: argoproj.io/v1alpha1
kind: ArgoCD
metadata:
  name: argocd
  namespace: argocd
spec:
  server:
    insecure: true
    service:
      type: LoadBalancer
  controller:
    resources:
      limits:
        cpu: 500m
        memory: 512Mi
      requests:
        cpu: 250m
        memory: 256Mi
  repoServer:
    resources:
      limits:
        cpu: 500m
        memory: 512Mi
      requests:
        cpu: 250m
        memory: 256Mi
  dex:
    resources:
      limits:
        cpu: 500m
        memory: 128Mi
      requests:
        cpu: 100m
        memory: 64Mi
  redis:
    resources:
      limits:
        cpu: 500m
        memory: 256Mi
      requests:
        cpu: 250m
        memory: 128Mi
```

### ArgoCD CLI Commands

```bash
# Install ArgoCD CLI
curl -sSL -o argocd-linux-amd64 https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64
sudo install -m 555 argocd-linux-amd64 /usr/local/bin/argocd
rm argocd-linux-amd64

# Login to ArgoCD
argocd login argocd.example.com

# Create application
argocd app create my-app \
  --repo https://github.com/your-org/your-repo.git \
  --path k8s/overlays/production \
  --dest-server https://kubernetes.default.svc \
  --dest-namespace production

# Sync application
argocd app sync my-app

# Get application status
argocd app get my-app

# List applications
argocd app list

# Delete application
argocd app delete my-app
```

## 🚀 Best Practices

### 1. GitOps Structure
```
k8s/
├── base/
│   ├── kustomization.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
├── overlays/
│   ├── development/
│   │   ├── kustomization.yaml
│   │   └── dev-patch.yaml
│   ├── staging/
│   │   ├── kustomization.yaml
│   │   └── staging-patch.yaml
│   └── production/
│       ├── kustomization.yaml
│       └── production-patch.yaml
└── argocd/
    ├── applications/
    └── app-of-apps.yaml
```

### 2. Security Best Practices
```yaml
# Use RBAC and secrets
apiVersion: v1
kind: Secret
metadata:
  name: argocd-secret
  namespace: argocd
type: Opaque
data:
  admin.password: <base64-encoded-password>
  admin.passwordMtime: <base64-encoded-timestamp>
```

### 3. Monitoring and Alerting
```yaml
# ArgoCD metrics
apiVersion: v1
kind: Service
metadata:
  name: argocd-metrics
  namespace: argocd
spec:
  selector:
    app.kubernetes.io/name: argocd-metrics
  ports:
  - port: 8083
    targetPort: 8083
```

## 🏢 Industry Insights

### ArgoCD Usage Patterns
- **GitOps**: Git as single source of truth
- **Multi-Environment**: Staging, production, development
- **Automated Sync**: Continuous deployment
- **Rollback**: Easy version management

### Enterprise ArgoCD Strategy
- **Multi-Cluster**: Cross-cluster deployment
- **Security**: RBAC and secrets management
- **Monitoring**: Application health monitoring
- **Compliance**: Audit trails and compliance

## 🎯 Interview Questions

### Basic Level
1. **What is ArgoCD?**
   - GitOps continuous delivery tool
   - Declarative deployment
   - Kubernetes-native
   - Git as source of truth

2. **What is GitOps?**
   - Git as single source of truth
   - Declarative configuration
   - Automated synchronization
   - Version control

3. **What are ArgoCD applications?**
   - Kubernetes resources
   - Git repository sources
   - Target clusters
   - Sync policies

### Intermediate Level
4. **How do you configure ArgoCD applications?**
   ```yaml
   apiVersion: argoproj.io/v1alpha1
   kind: Application
   metadata:
     name: my-app
   spec:
     source:
       repoURL: https://github.com/org/repo.git
       path: k8s/overlays/production
     destination:
       server: https://kubernetes.default.svc
       namespace: production
   ```

5. **How do you handle ArgoCD security?**
   - RBAC configuration
   - Secret management
   - Network policies
   - Access controls

6. **How do you implement ArgoCD monitoring?**
   - Application health checks
   - Sync status monitoring
   - Performance metrics
   - Alerting

### Advanced Level
7. **How do you implement ArgoCD patterns?**
   - Application sets
   - App of apps
   - Multi-cluster deployment
   - Progressive delivery

8. **How do you handle ArgoCD scaling?**
   - Multi-cluster management
   - Resource optimization
   - Performance tuning
   - Load balancing

9. **How do you implement ArgoCD testing?**
   - Canary deployments
   - Blue-green deployments
   - A/B testing
   - Rollback strategies

---

**Next**: [Containers](./Containers/) - Docker, Kubernetes, container orchestration
