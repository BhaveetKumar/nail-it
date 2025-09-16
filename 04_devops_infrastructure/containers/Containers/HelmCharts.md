# 📦 Helm Charts: Package Management and Templating

> **Master Helm for Kubernetes package management, templating, and deployment**

## 📚 Concept

**Detailed Explanation:**
Helm is the de facto package manager for Kubernetes that simplifies the deployment and management of complex applications. It provides a powerful templating system, dependency management, and release lifecycle management for Kubernetes applications. Helm transforms the complexity of managing multiple Kubernetes resources into a simple, declarative approach using charts.

**Core Philosophy:**

- **Package Management**: Treat Kubernetes applications as packages with dependencies
- **Templating**: Use dynamic configuration to adapt to different environments
- **Versioning**: Maintain version history and enable easy rollbacks
- **Dependency Management**: Handle complex application dependencies automatically
- **Release Management**: Track and manage application deployments as releases
- **Lifecycle Hooks**: Execute custom logic during deployment lifecycle

**Why Helm Matters:**

- **Complexity Reduction**: Simplify deployment of multi-resource applications
- **Environment Management**: Easily deploy to different environments with configuration
- **Dependency Resolution**: Automatically handle application dependencies
- **Rollback Capability**: Quickly revert to previous working versions
- **Template Reusability**: Create reusable templates for common patterns
- **Release Tracking**: Maintain history of all deployments and changes
- **Community Ecosystem**: Access to thousands of pre-built charts
- **Enterprise Features**: Support for private repositories and security

**Key Features:**

**1. Package Management:**

- **Charts**: Packages containing Kubernetes resource templates
- **Repositories**: Collections of charts for easy discovery and installation
- **Dependencies**: Automatic resolution and installation of chart dependencies
- **Versioning**: Semantic versioning for charts and releases
- **Benefits**: Simplified application deployment, dependency management, version control
- **Use Cases**: Complex applications, microservices, database deployments

**2. Templating:**

- **Go Templates**: Powerful templating engine based on Go templates
- **Dynamic Configuration**: Generate Kubernetes manifests based on values
- **Conditional Logic**: Include/exclude resources based on configuration
- **Helper Functions**: Built-in functions for common operations
- **Benefits**: Single template for multiple environments, reduced duplication
- **Use Cases**: Environment-specific deployments, configuration management

**3. Versioning:**

- **Chart Versions**: Semantic versioning for chart packages
- **Release Versions**: Track different deployments of the same chart
- **Release History**: Maintain complete history of all deployments
- **Rollback**: Easy rollback to any previous release version
- **Benefits**: Version control, rollback capability, deployment tracking
- **Use Cases**: Production deployments, A/B testing, disaster recovery

**4. Dependencies:**

- **Chart Dependencies**: Define dependencies on other charts
- **Repository Management**: Manage multiple chart repositories
- **Dependency Resolution**: Automatic resolution of dependency conflicts
- **Conditional Dependencies**: Enable/disable dependencies based on configuration
- **Benefits**: Modular architecture, reusable components, simplified management
- **Use Cases**: Microservices, shared infrastructure, database dependencies

**5. Rollback:**

- **Release History**: Maintain complete history of all releases
- **One-Command Rollback**: Simple command to revert to previous version
- **Selective Rollback**: Rollback specific resources or entire release
- **Rollback Validation**: Validate rollback before execution
- **Benefits**: Quick recovery, risk mitigation, deployment confidence
- **Use Cases**: Production issues, failed deployments, emergency recovery

**6. Hooks:**

- **Lifecycle Hooks**: Execute custom logic at specific deployment phases
- **Hook Types**: Pre-install, post-install, pre-upgrade, post-upgrade, pre-delete, post-delete
- **Job Resources**: Use Kubernetes Jobs for hook execution
- **Hook Management**: Automatic cleanup and management of hook resources
- **Benefits**: Custom deployment logic, database migrations, validation
- **Use Cases**: Database migrations, service validation, custom deployment steps

**Advanced Helm Concepts:**

- **Chart Libraries**: Reusable chart components and functions
- **Sub-charts**: Charts that are dependencies of other charts
- **Chart Tests**: Automated testing of chart functionality
- **Chart Signing**: Cryptographic signing of charts for security
- **Multi-Environment**: Deploy same chart to multiple environments
- **Custom Resources**: Support for Kubernetes custom resources
- **Plugin System**: Extend Helm functionality with plugins

**Discussion Questions & Answers:**

**Q1: How do you design a comprehensive Helm chart architecture for a large-scale microservices application with complex dependencies and multiple environments?**

**Answer:** Comprehensive Helm chart architecture design:

- **Chart Organization**: Create separate charts for each microservice with shared library charts
- **Dependency Management**: Use chart dependencies for shared infrastructure (databases, monitoring)
- **Environment Configuration**: Implement environment-specific values files with inheritance
- **Template Structure**: Use helper templates and partials for reusable components
- **Security**: Implement proper secret management and RBAC configurations
- **Testing**: Create comprehensive chart tests for validation and regression testing
- **Documentation**: Maintain detailed documentation for chart usage and configuration
- **Versioning**: Implement semantic versioning and release management
- **CI/CD Integration**: Integrate with CI/CD pipelines for automated testing and deployment
- **Monitoring**: Implement health checks and monitoring configurations
- **Rollback Strategy**: Design comprehensive rollback procedures and testing
- **Governance**: Establish chart review and approval processes

**Q2: What are the key considerations when implementing Helm for a production environment with strict security and compliance requirements?**

**Answer:** Production Helm security and compliance implementation:

- **Chart Security**: Implement chart signing and verification for all charts
- **Repository Security**: Use private repositories with proper access controls
- **Secret Management**: Implement secure secret management using external secret operators
- **RBAC Integration**: Configure proper RBAC for Helm and Kubernetes resources
- **Network Policies**: Implement network policies for secure communication
- **Compliance**: Ensure charts meet regulatory and compliance requirements
- **Audit Logging**: Implement comprehensive audit logging for all Helm operations
- **Backup and Recovery**: Implement proper backup and recovery procedures
- **Testing**: Conduct security testing and vulnerability scanning
- **Documentation**: Maintain security documentation and procedures
- **Training**: Provide security training for teams using Helm
- **Monitoring**: Implement security monitoring and alerting

**Q3: How do you optimize Helm deployments for performance, reliability, and maintainability in enterprise environments?**

**Answer:** Enterprise Helm optimization strategies:

- **Chart Performance**: Optimize chart templates for faster rendering and deployment
- **Dependency Optimization**: Minimize dependencies and use efficient dependency resolution
- **Resource Management**: Implement proper resource limits and requests
- **Deployment Strategy**: Use blue-green or canary deployments for zero-downtime updates
- **Monitoring**: Implement comprehensive monitoring and alerting for deployments
- **Error Handling**: Implement robust error handling and retry mechanisms
- **Documentation**: Maintain comprehensive documentation and runbooks
- **Automation**: Implement automated testing and deployment pipelines
- **Governance**: Establish governance processes for chart development and deployment
- **Training**: Provide training for teams on Helm best practices
- **Performance Testing**: Conduct performance testing of chart deployments
- **Regular Reviews**: Conduct regular reviews of chart performance and optimization

## 🏗️ Helm Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Helm Client                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Chart     │  │   Template  │  │   Values    │     │
│  │ Repository  │  │   Engine    │  │   Files     │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Tiller (Helm 2)                      │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Release   │  │   Template  │  │   Config    │ │ │
│  │  │ Management  │  │   Rendering │  │   Storage   │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Kubernetes  │  │ Kubernetes  │  │ Kubernetes  │     │
│  │   Cluster   │  │   Cluster   │  │   Cluster   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### Basic Chart Structure

```
my-app-chart/
├── Chart.yaml
├── values.yaml
├── values-dev.yaml
├── values-prod.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── ingress.yaml
│   ├── hpa.yaml
│   ├── _helpers.tpl
│   └── NOTES.txt
├── charts/
└── README.md
```

### Chart.yaml

```yaml
# Chart.yaml
apiVersion: v2
name: my-app-chart
description: A Helm chart for My Application
type: application
version: 1.0.0
appVersion: "1.0.0"
home: https://github.com/your-org/my-app
sources:
  - https://github.com/your-org/my-app
maintainers:
  - name: Your Name
    email: your.email@example.com
keywords:
  - web
  - api
  - microservice
dependencies:
  - name: postgresql
    version: 12.1.2
    repository: https://charts.bitnami.com/bitnami
    condition: postgresql.enabled
  - name: redis
    version: 17.3.7
    repository: https://charts.bitnami.com/bitnami
    condition: redis.enabled
```

### values.yaml

```yaml
# values.yaml
# Default values for my-app-chart
replicaCount: 3

image:
  repository: my-registry.com/my-app
  pullPolicy: IfNotPresent
  tag: "latest"

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

podAnnotations: {}

podSecurityContext:
  fsGroup: 1000

securityContext:
  capabilities:
    drop:
      - ALL
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  runAsUser: 1000

service:
  type: ClusterIP
  port: 80
  targetPort: 8080

ingress:
  enabled: false
  className: ""
  annotations: {}
  hosts:
    - host: my-app.example.com
      paths:
        - path: /
          pathType: Prefix
  tls: []

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 250m
    memory: 256Mi

autoscaling:
  enabled: false
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}

tolerations: []

affinity: {}

# Application configuration
app:
  env: production
  logLevel: info
  database:
    host: postgresql
    port: 5432
    name: myapp
    username: postgres
    password: ""
  redis:
    host: redis
    port: 6379
    password: ""

# Dependencies
postgresql:
  enabled: true
  auth:
    postgresPassword: "postgres"
    database: "myapp"
  primary:
    persistence:
      enabled: true
      size: 8Gi

redis:
  enabled: true
  auth:
    enabled: false
  master:
    persistence:
      enabled: true
      size: 1Gi
```

### Deployment Template

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "my-app-chart.fullname" . }}
  labels:
    {{- include "my-app-chart.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "my-app-chart.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      {{- with .Values.podAnnotations }}
      annotations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      labels:
        {{- include "my-app-chart.selectorLabels" . | nindent 8 }}
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "my-app-chart.serviceAccountName" . }}
      securityContext:
        {{- toYaml .Values.podSecurityContext | nindent 8 }}
      containers:
        - name: {{ .Chart.Name }}
          securityContext:
            {{- toYaml .Values.securityContext | nindent 12 }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: {{ .Values.service.targetPort }}
              protocol: TCP
          env:
            - name: ENV
              value: {{ .Values.app.env | quote }}
            - name: LOG_LEVEL
              value: {{ .Values.app.logLevel | quote }}
            - name: DATABASE_HOST
              value: {{ .Values.app.database.host | quote }}
            - name: DATABASE_PORT
              value: {{ .Values.app.database.port | quote }}
            - name: DATABASE_NAME
              value: {{ .Values.app.database.name | quote }}
            - name: DATABASE_USERNAME
              value: {{ .Values.app.database.username | quote }}
            - name: DATABASE_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: {{ include "my-app-chart.fullname" . }}-secret
                  key: database-password
            - name: REDIS_HOST
              value: {{ .Values.app.redis.host | quote }}
            - name: REDIS_PORT
              value: {{ .Values.app.redis.port | quote }}
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
```

### Service Template

```yaml
# templates/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: { { include "my-app-chart.fullname" . } }
  labels: { { - include "my-app-chart.labels" . | nindent 4 } }
spec:
  type: { { .Values.service.type } }
  ports:
    - port: { { .Values.service.port } }
      targetPort: { { .Values.service.targetPort } }
      protocol: TCP
      name: http
  selector: { { - include "my-app-chart.selectorLabels" . | nindent 4 } }
```

### ConfigMap Template

```yaml
# templates/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "my-app-chart.fullname" . }}-config
  labels:
    {{- include "my-app-chart.labels" . | nindent 4 }}
data:
  app.properties: |
    server.port={{ .Values.service.targetPort }}
    logging.level={{ .Values.app.logLevel }}
    database.pool.size=10
    redis.timeout=5000
  nginx.conf: |
    server {
        listen 80;
        location / {
            proxy_pass http://{{ include "my-app-chart.fullname" . }}:{{ .Values.service.port }};
        }
    }
```

### Secret Template

```yaml
# templates/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: {{ include "my-app-chart.fullname" . }}-secret
  labels:
    {{- include "my-app-chart.labels" . | nindent 4 }}
type: Opaque
data:
  database-password: {{ .Values.postgresql.auth.postgresPassword | b64enc | quote }}
  redis-password: {{ .Values.redis.auth.password | default "" | b64enc | quote }}
```

### Ingress Template

```yaml
# templates/ingress.yaml
{{- if .Values.ingress.enabled -}}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ include "my-app-chart.fullname" . }}
  labels:
    {{- include "my-app-chart.labels" . | nindent 4 }}
  {{- with .Values.ingress.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  {{- if .Values.ingress.className }}
  ingressClassName: {{ .Values.ingress.className }}
  {{- end }}
  {{- if .Values.ingress.tls }}
  tls:
    {{- range .Values.ingress.tls }}
    - hosts:
        {{- range .hosts }}
        - {{ . | quote }}
        {{- end }}
      secretName: {{ .secretName }}
    {{- end }}
  {{- end }}
  rules:
    {{- range .Values.ingress.hosts }}
    - host: {{ .host | quote }}
      http:
        paths:
          {{- range .paths }}
          - path: {{ .path }}
            pathType: {{ .pathType }}
            backend:
              service:
                name: {{ include "my-app-chart.fullname" $ }}
                port:
                  number: {{ $.Values.service.port }}
          {{- end }}
    {{- end }}
{{- end }}
```

### HPA Template

```yaml
# templates/hpa.yaml
{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "my-app-chart.fullname" . }}
  labels:
    {{- include "my-app-chart.labels" . | nindent 4 }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "my-app-chart.fullname" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    {{- if .Values.autoscaling.targetCPUUtilizationPercentage }}
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
    {{- end }}
    {{- if .Values.autoscaling.targetMemoryUtilizationPercentage }}
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetMemoryUtilizationPercentage }}
    {{- end }}
{{- end }}
```

### Helpers Template

```yaml
# templates/_helpers.tpl
{{/*
Expand the name of the chart.
*/}}
{{- define "my-app-chart.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "my-app-chart.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "my-app-chart.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "my-app-chart.labels" -}}
helm.sh/chart: {{ include "my-app-chart.chart" . }}
{{ include "my-app-chart.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "my-app-chart.selectorLabels" -}}
app.kubernetes.io/name: {{ include "my-app-chart.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "my-app-chart.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "my-app-chart.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
```

### Environment-Specific Values

```yaml
# values-dev.yaml
replicaCount: 1

image:
  tag: "dev"

app:
  env: development
  logLevel: debug

postgresql:
  enabled: true
  auth:
    postgresPassword: "devpassword"
  primary:
    persistence:
      enabled: false

redis:
  enabled: true
  auth:
    enabled: false
  master:
    persistence:
      enabled: false

ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: my-app-dev.example.com
      paths:
        - path: /
          pathType: Prefix
```

```yaml
# values-prod.yaml
replicaCount: 5

image:
  tag: "v1.0.0"

app:
  env: production
  logLevel: info

postgresql:
  enabled: true
  auth:
    postgresPassword: "securepassword"
  primary:
    persistence:
      enabled: true
      size: 50Gi

redis:
  enabled: true
  auth:
    enabled: true
    password: "redispassword"
  master:
    persistence:
      enabled: true
      size: 10Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    - host: my-app.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: my-app-tls
      hosts:
        - my-app.example.com
```

### Helm Commands

```bash
# Create a new chart
helm create my-app-chart

# Install chart
helm install my-app ./my-app-chart
helm install my-app ./my-app-chart -f values-dev.yaml
helm install my-app ./my-app-chart --set replicaCount=5

# Upgrade chart
helm upgrade my-app ./my-app-chart
helm upgrade my-app ./my-app-chart -f values-prod.yaml

# Rollback chart
helm rollback my-app 1

# List releases
helm list
helm list --all-namespaces

# Get release status
helm status my-app

# Get release history
helm history my-app

# Uninstall chart
helm uninstall my-app

# Package chart
helm package ./my-app-chart

# Add repository
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Install from repository
helm install postgresql bitnami/postgresql

# Template chart
helm template my-app ./my-app-chart
helm template my-app ./my-app-chart -f values-prod.yaml

# Lint chart
helm lint ./my-app-chart

# Test chart
helm test my-app

# Dependency management
helm dependency update ./my-app-chart
helm dependency build ./my-app-chart
```

## 🚀 Best Practices

### 1. Chart Structure

```
my-app-chart/
├── Chart.yaml          # Chart metadata
├── values.yaml         # Default values
├── values-*.yaml       # Environment-specific values
├── templates/          # Kubernetes templates
│   ├── _helpers.tpl    # Template helpers
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ...
├── charts/             # Chart dependencies
└── README.md           # Documentation
```

### 2. Template Best Practices

```yaml
# Use helpers for common patterns
{{- include "my-app-chart.fullname" . }}

# Use conditionals for optional features
{{- if .Values.ingress.enabled }}
# ingress template
{{- end }}

# Use with blocks for scoping
{{- with .Values.resources }}
resources:
  {{- toYaml . | nindent 2 }}
{{- end }}
```

### 3. Values Management

```yaml
# Use environment-specific values files
helm install my-app ./my-app-chart -f values-dev.yaml
helm install my-app ./my-app-chart -f values-prod.yaml

# Use --set for overrides
helm install my-app ./my-app-chart --set replicaCount=5
```

## 🏢 Industry Insights

### Helm Usage Patterns

- **Package Management**: Kubernetes application packaging
- **Templating**: Dynamic configuration
- **Versioning**: Release management
- **Dependencies**: Chart dependencies

### Enterprise Helm Strategy

- **Private Repositories**: Internal chart repositories
- **Security**: Chart signing and verification
- **Governance**: Chart review and approval
- **Automation**: CI/CD integration

## 🎯 Interview Questions

### Basic Level

1. **What is Helm?**

   - Kubernetes package manager
   - Chart templating
   - Release management
   - Dependency management

2. **What are Helm charts?**

   - Kubernetes application packages
   - Template collections
   - Versioned releases
   - Dependency definitions

3. **What are Helm values?**
   - Configuration parameters
   - Template variables
   - Environment settings
   - Override mechanisms

### Intermediate Level

4. **How do you create Helm charts?**

   ```bash
   # Create new chart
   helm create my-app-chart

   # Customize templates
   # Edit values.yaml
   # Add dependencies
   ```

5. **How do you handle Helm templating?**

   - Go template syntax
   - Helper functions
   - Conditional logic
   - Value substitution

6. **How do you manage Helm dependencies?**
   - Chart.yaml dependencies
   - Repository management
   - Version constraints
   - Update strategies

### Advanced Level

7. **How do you implement Helm patterns?**

   - Chart libraries
   - Sub-charts
   - Hooks
   - Tests

8. **How do you handle Helm security?**

   - Chart signing
   - Repository security
   - RBAC integration
   - Secret management

9. **How do you implement Helm automation?**
   - CI/CD integration
   - Automated testing
   - Release automation
   - Rollback strategies

---

**Next**: [Service Mesh - Istio](ServiceMesh_Istio.md/) - Microservices communication, security, observability
