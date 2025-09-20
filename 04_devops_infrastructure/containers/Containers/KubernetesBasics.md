# ☸️ Kubernetes Basics: Container Orchestration and Scaling

> **Master Kubernetes for container orchestration, scaling, and management**

## 📚 Concept

**Detailed Explanation:**
Kubernetes (K8s) is an open-source container orchestration platform that automates the deployment, scaling, and management of containerized applications. Originally developed by Google and now maintained by the Cloud Native Computing Foundation (CNCF), Kubernetes provides a robust framework for running distributed systems resiliently across multiple machines.

**Why Kubernetes Matters:**

- **Container Orchestration**: Manages complex containerized applications across multiple hosts
- **Scalability**: Automatically scales applications based on demand
- **High Availability**: Ensures applications remain available even when individual components fail
- **Portability**: Runs consistently across on-premises, hybrid, and multi-cloud environments
- **Ecosystem**: Rich ecosystem of tools and extensions for various use cases
- **Industry Standard**: De facto standard for container orchestration

**Core Philosophy:**

- **Declarative Configuration**: Describe desired state, let Kubernetes make it happen
- **Self-Healing**: Automatically restarts failed containers and replaces unhealthy nodes
- **Automated Rollouts**: Zero-downtime deployments with rollback capabilities
- **Resource Management**: Efficient allocation and utilization of compute resources
- **Extensibility**: Pluggable architecture for custom resources and controllers

### Key Features

**Detailed Explanation:**
Kubernetes provides a comprehensive set of features that make it suitable for managing complex, distributed applications in production environments.

**Container Orchestration:**

- **Pod Management**: Groups containers that need to work together
- **Scheduling**: Intelligently places pods on available nodes
- **Lifecycle Management**: Handles container startup, shutdown, and restart
- **Resource Allocation**: Manages CPU, memory, and storage resources
- **Multi-Container Coordination**: Orchestrates complex multi-container applications

**Scaling:**

- **Horizontal Scaling**: Automatically adds or removes pod replicas based on demand
- **Vertical Scaling**: Adjusts resource limits for individual containers
- **Predictive Scaling**: Uses machine learning to predict scaling needs
- **Custom Metrics**: Scales based on application-specific metrics
- **Scheduled Scaling**: Scales based on predictable load patterns

**Service Discovery:**

- **DNS-Based Discovery**: Automatic DNS resolution for services
- **Environment Variables**: Injects service endpoints as environment variables
- **Service Mesh Integration**: Works with service mesh solutions for advanced networking
- **Cross-Namespace Discovery**: Services can discover each other across namespaces
- **External Service Integration**: Connects to services outside the cluster

**Load Balancing:**

- **Service Load Balancing**: Distributes traffic across multiple pod replicas
- **Session Affinity**: Maintains sticky sessions when needed
- **Health Check Integration**: Routes traffic only to healthy pods
- **External Load Balancers**: Integrates with cloud provider load balancers
- **Ingress Controllers**: Advanced routing and load balancing at the HTTP/HTTPS level

**Self-Healing:**

- **Pod Restart**: Automatically restarts failed containers
- **Node Replacement**: Replaces pods on failed nodes
- **Health Monitoring**: Continuously monitors application health
- **Automatic Recovery**: Recovers from various failure scenarios
- **Circuit Breaker Patterns**: Implements resilience patterns

**Rolling Updates:**

- **Zero-Downtime Deployments**: Updates applications without service interruption
- **Rollback Capability**: Quickly reverts to previous versions if issues occur
- **Canary Deployments**: Gradually roll out changes to a subset of users
- **Blue-Green Deployments**: Maintains two identical environments for switching
- **A/B Testing**: Supports testing different versions with different user groups

**Discussion Questions & Answers:**

**Q1: How does Kubernetes compare to other container orchestration platforms?**

**Answer:** Kubernetes advantages:

- **Ecosystem**: Largest ecosystem with extensive tooling and community support
- **Portability**: Runs on any cloud provider or on-premises infrastructure
- **Maturity**: Battle-tested in production environments at scale
- **Flexibility**: Highly configurable and extensible architecture
- **Industry Adoption**: Widely adopted by enterprises and cloud providers
- **Standards**: De facto standard for container orchestration

**Q2: What are the key challenges when adopting Kubernetes?**

**Answer:** Common challenges include:

- **Complexity**: Steep learning curve for teams new to container orchestration
- **Operational Overhead**: Requires dedicated DevOps expertise for management
- **Resource Requirements**: Needs significant compute resources for the control plane
- **Security**: Complex security model requiring careful configuration
- **Networking**: Advanced networking concepts and configuration
- **Storage**: Persistent storage management can be complex
- **Monitoring**: Requires comprehensive observability stack

**Q3: How do you decide when to use Kubernetes vs simpler alternatives?**

**Answer:** Use Kubernetes when:

- **Scale**: Need to manage hundreds or thousands of containers
- **Complexity**: Have complex microservices architectures
- **High Availability**: Require high availability and fault tolerance
- **Multi-Environment**: Need consistent deployments across environments
- **Team Size**: Have dedicated DevOps/Platform teams
- **Future Growth**: Expect significant growth in containerized workloads

Use simpler alternatives when:

- **Small Scale**: Managing only a few containers
- **Simple Applications**: Single-service or simple multi-service applications
- **Limited Resources**: Don't have resources for Kubernetes management
- **Learning Curve**: Team lacks experience with container orchestration
- **Quick Start**: Need to get started quickly without complex setup

## 🏗️ Kubernetes Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Master    │  │   Master    │  │   Master    │     │
│  │   Node      │  │   Node      │  │   Node      │     │
│  │  (Control   │  │  (Control   │  │  (Control   │     │
│  │   Plane)    │  │   Plane)    │  │   Plane)    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Worker Nodes                          │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Worker    │  │   Worker    │  │   Worker    │ │ │
│  │  │   Node      │  │   Node      │  │   Node      │ │ │
│  │  │  (Pods)     │  │  (Pods)     │  │  (Pods)     │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   etcd      │  │   CNI       │  │   CRI       │     │
│  │   Storage   │  │   Network   │  │   Runtime   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Hands-on Example

### Basic Pod Configuration

```yaml
# pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app-pod
  labels:
    app: my-app
    version: v1.0.0
spec:
  containers:
    - name: my-app
      image: my-registry.com/my-app:latest
      ports:
        - containerPort: 8080
          protocol: TCP
      env:
        - name: DATABASE_URL
          value: "postgresql://user:password@db:5432/mydb"
        - name: LOG_LEVEL
          value: "info"
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
      readinessProbe:
        httpGet:
          path: /ready
          port: 8080
        initialDelaySeconds: 5
        periodSeconds: 5
      volumeMounts:
        - name: config-volume
          mountPath: /app/config
  volumes:
    - name: config-volume
      configMap:
        name: my-app-config
  restartPolicy: Always
```

### Deployment Configuration

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app-deployment
  labels:
    app: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
        version: v1.0.0
    spec:
      containers:
        - name: my-app
          image: my-registry.com/my-app:latest
          ports:
            - containerPort: 8080
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: my-app-secrets
                  key: database-url
            - name: LOG_LEVEL
              value: "info"
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
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
          volumeMounts:
            - name: config-volume
              mountPath: /app/config
      volumes:
        - name: config-volume
          configMap:
            name: my-app-config
      restartPolicy: Always
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
```

### Service Configuration

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
  labels:
    app: my-app
spec:
  selector:
    app: my-app
  ports:
    - name: http
      port: 80
      targetPort: 8080
      protocol: TCP
  type: ClusterIP
---
apiVersion: v1
kind: Service
metadata:
  name: my-app-service-lb
  labels:
    app: my-app
spec:
  selector:
    app: my-app
  ports:
    - name: http
      port: 80
      targetPort: 8080
      protocol: TCP
  type: LoadBalancer
```

### ConfigMap and Secret

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-app-config
data:
  app.properties: |
    server.port=8080
    logging.level=INFO
    database.pool.size=10
  nginx.conf: |
    server {
        listen 80;
        location / {
            proxy_pass http://my-app-service:80;
        }
    }
---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: my-app-secrets
type: Opaque
data:
  database-url: cG9zdGdyZXNxbDovL3VzZXI6cGFzc3dvcmRAZGI6NTQzMi9teWRi
  api-key: YWJjZGVmZ2hpams=
```

### Ingress Configuration

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-app-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
    - hosts:
        - my-app.example.com
      secretName: my-app-tls
  rules:
    - host: my-app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: my-app-service
                port:
                  number: 80
```

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app-deployment
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
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
        - type: Pods
          value: 4
          periodSeconds: 15
      selectPolicy: Max
```

### Namespace and RBAC

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: my-app-namespace
  labels:
    name: my-app-namespace
---
# rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: my-app-sa
  namespace: my-app-namespace
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: my-app-namespace
  name: my-app-role
rules:
  - apiGroups: [""]
    resources: ["pods", "services", "configmaps", "secrets"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: ["apps"]
    resources: ["deployments", "replicasets"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: my-app-rolebinding
  namespace: my-app-namespace
subjects:
  - kind: ServiceAccount
    name: my-app-sa
    namespace: my-app-namespace
roleRef:
  kind: Role
  name: my-app-role
  apiGroup: rbac.authorization.k8s.io
```

### Kubernetes Commands

```bash
# Create resources
kubectl apply -f pod.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

# Get resources
kubectl get pods
kubectl get deployments
kubectl get services
kubectl get configmaps
kubectl get secrets
kubectl get ingress
kubectl get hpa

# Describe resources
kubectl describe pod my-app-pod
kubectl describe deployment my-app-deployment
kubectl describe service my-app-service

# View logs
kubectl logs my-app-pod
kubectl logs -f my-app-pod
kubectl logs deployment/my-app-deployment

# Execute commands
kubectl exec -it my-app-pod -- /bin/bash
kubectl exec -it my-app-pod -- curl localhost:8080/health

# Scale deployments
kubectl scale deployment my-app-deployment --replicas=5
kubectl autoscale deployment my-app-deployment --cpu-percent=70 --min=2 --max=10

# Update deployments
kubectl set image deployment/my-app-deployment my-app=my-registry.com/my-app:v2.0.0
kubectl rollout status deployment/my-app-deployment
kubectl rollout history deployment/my-app-deployment
kubectl rollout undo deployment/my-app-deployment

# Port forwarding
kubectl port-forward pod/my-app-pod 8080:8080
kubectl port-forward service/my-app-service 8080:80

# Delete resources
kubectl delete pod my-app-pod
kubectl delete deployment my-app-deployment
kubectl delete service my-app-service
kubectl delete configmap my-app-config
kubectl delete secret my-app-secrets
kubectl delete ingress my-app-ingress
kubectl delete hpa my-app-hpa
```

## 🚀 Best Practices

### 1. Resource Management

```yaml
# Set resource requests and limits
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

### 2. Health Checks

```yaml
# Implement liveness and readiness probes
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
```

### 3. Security Best Practices

```yaml
# Use non-root user
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
```

## 🏢 Industry Insights

### Kubernetes Usage Patterns

- **Microservices**: Container orchestration
- **Scaling**: Horizontal and vertical scaling
- **CI/CD**: Automated deployments
- **Multi-Cloud**: Cross-cloud deployments

### Enterprise Kubernetes Strategy

- **Security**: RBAC and network policies
- **Monitoring**: Observability and logging
- **Backup**: Disaster recovery
- **Compliance**: Audit trails and compliance

## 🎯 Interview Questions

### Basic Level

1. **What is Kubernetes?**

   - Container orchestration platform
   - Automated deployment
   - Scaling and management
   - Self-healing

2. **What are Kubernetes pods?**

   - Smallest deployable unit
   - Container wrapper
   - Shared networking
   - Shared storage

3. **What are Kubernetes services?**
   - Service discovery
   - Load balancing
   - Network abstraction
   - Stable endpoints

### Intermediate Level

4. **How do you implement Kubernetes scaling?**

   ```yaml
   # Horizontal Pod Autoscaler
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   spec:
     minReplicas: 2
     maxReplicas: 10
     metrics:
       - type: Resource
         resource:
           name: cpu
           target:
             averageUtilization: 70
   ```

5. **How do you handle Kubernetes security?**

   - RBAC configuration
   - Network policies
   - Pod security policies
   - Secrets management

6. **How do you implement Kubernetes monitoring?**
   - Health checks
   - Metrics collection
   - Log aggregation
   - Alerting

### Advanced Level

7. **How do you implement Kubernetes patterns?**

   - Sidecar pattern
   - Init containers
   - Job and CronJob
   - StatefulSets

8. **How do you handle Kubernetes networking?**

   - Service mesh
   - Ingress controllers
   - Network policies
   - Load balancing

9. **How do you implement Kubernetes storage?**
   - Persistent volumes
   - Storage classes
   - Volume claims
   - Backup strategies

---

**Next**: [Helm Charts](HelmCharts.md) - Package management, templating, deployment
