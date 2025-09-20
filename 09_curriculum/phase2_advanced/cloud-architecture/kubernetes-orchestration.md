# Kubernetes Orchestration

## Overview

This module covers Kubernetes orchestration concepts including pod management, service discovery, deployment strategies, and cluster management. These concepts are essential for container orchestration and cloud-native applications.

## Table of Contents

1. [Pod Management](#pod-management)
2. [Service Discovery](#service-discovery)
3. [Deployment Strategies](#deployment-strategies)
4. [Cluster Management](#cluster-management)
5. [Applications](#applications)
6. [Complexity Analysis](#complexity-analysis)
7. [Follow-up Questions](#follow-up-questions)

## Pod Management

### Theory

Pods are the smallest deployable units in Kubernetes. They contain one or more containers and share storage and network resources. Pod management involves creating, scheduling, and monitoring pods.

### Pod Management Implementation

#### Golang Implementation

```go
package main

import (
    "encoding/json"
    "fmt"
    "time"
)

type Pod struct {
    Name        string
    Namespace   string
    Labels      map[string]string
    Containers  []Container
    Status      string
    Node        string
    CreatedAt   time.Time
    UpdatedAt   time.Time
}

type Container struct {
    Name      string
    Image     string
    Port      int
    Resources ResourceRequirements
    Status    string
}

type ResourceRequirements struct {
    CPU    string
    Memory string
    Limits ResourceLimits
}

type ResourceLimits struct {
    CPU    string
    Memory string
}

type PodManager struct {
    Pods map[string]*Pod
    mutex sync.RWMutex
}

func NewPodManager() *PodManager {
    return &PodManager{
        Pods: make(map[string]*Pod),
    }
}

func (pm *PodManager) CreatePod(name, namespace string, containers []Container, labels map[string]string) *Pod {
    pod := &Pod{
        Name:       name,
        Namespace:  namespace,
        Labels:     labels,
        Containers: containers,
        Status:     "Pending",
        CreatedAt:  time.Now(),
        UpdatedAt:  time.Now(),
    }
    
    pm.mutex.Lock()
    pm.Pods[name] = pod
    pm.mutex.Unlock()
    
    fmt.Printf("Created pod: %s in namespace %s\n", name, namespace)
    return pod
}

func (pm *PodManager) GetPod(name string) *Pod {
    pm.mutex.RLock()
    defer pm.mutex.RUnlock()
    
    return pm.Pods[name]
}

func (pm *PodManager) ListPods(namespace string) []*Pod {
    pm.mutex.RLock()
    defer pm.mutex.RUnlock()
    
    var pods []*Pod
    for _, pod := range pm.Pods {
        if namespace == "" || pod.Namespace == namespace {
            pods = append(pods, pod)
        }
    }
    
    return pods
}

func (pm *PodManager) UpdatePodStatus(name, status string) {
    pm.mutex.Lock()
    defer pm.mutex.Unlock()
    
    if pod, exists := pm.Pods[name]; exists {
        pod.Status = status
        pod.UpdatedAt = time.Now()
        fmt.Printf("Updated pod %s status to: %s\n", name, status)
    }
}

func (pm *PodManager) SchedulePod(name, node string) {
    pm.mutex.Lock()
    defer pm.mutex.Unlock()
    
    if pod, exists := pm.Pods[name]; exists {
        pod.Node = node
        pod.Status = "Running"
        pod.UpdatedAt = time.Now()
        fmt.Printf("Scheduled pod %s to node %s\n", name, node)
    }
}

func (pm *PodManager) DeletePod(name string) bool {
    pm.mutex.Lock()
    defer pm.mutex.Unlock()
    
    if _, exists := pm.Pods[name]; exists {
        delete(pm.Pods, name)
        fmt.Printf("Deleted pod: %s\n", name)
        return true
    }
    
    return false
}

func (pm *PodManager) GetPodLogs(name string) []string {
    pm.mutex.RLock()
    pod, exists := pm.Pods[name]
    pm.mutex.RUnlock()
    
    if !exists {
        return []string{}
    }
    
    // Simulate pod logs
    logs := []string{
        fmt.Sprintf("Pod %s started", name),
        fmt.Sprintf("Container %s running", pod.Containers[0].Name),
        fmt.Sprintf("Health check passed"),
    }
    
    return logs
}

func (pm *PodManager) GetPodMetrics(name string) map[string]interface{} {
    pm.mutex.RLock()
    pod, exists := pm.Pods[name]
    pm.mutex.RUnlock()
    
    if !exists {
        return map[string]interface{}{}
    }
    
    return map[string]interface{}{
        "name":      pod.Name,
        "namespace": pod.Namespace,
        "status":    pod.Status,
        "node":      pod.Node,
        "cpu_usage": "150m",
        "memory_usage": "256Mi",
        "uptime":    time.Since(pod.CreatedAt).String(),
    }
}

func (pm *PodManager) ScalePod(name string, replicas int) {
    pm.mutex.Lock()
    defer pm.mutex.Unlock()
    
    if pod, exists := pm.Pods[name]; exists {
        // In a real implementation, this would create multiple pod instances
        fmt.Printf("Scaling pod %s to %d replicas\n", name, replicas)
        pod.UpdatedAt = time.Now()
    }
}

func (pm *PodManager) GetPodEvents(name string) []string {
    events := []string{
        fmt.Sprintf("Pod %s created", name),
        fmt.Sprintf("Pod %s scheduled", name),
        fmt.Sprintf("Pod %s started", name),
        fmt.Sprintf("Pod %s ready", name),
    }
    
    return events
}

func main() {
    pm := NewPodManager()
    
    fmt.Println("Kubernetes Pod Management Demo:")
    
    // Create containers
    containers := []Container{
        {
            Name:  "nginx",
            Image: "nginx:1.20",
            Port:  80,
            Resources: ResourceRequirements{
                CPU:    "100m",
                Memory: "128Mi",
                Limits: ResourceLimits{
                    CPU:    "200m",
                    Memory: "256Mi",
                },
            },
            Status: "Running",
        },
    }
    
    labels := map[string]string{
        "app":     "web",
        "version": "1.0",
        "tier":    "frontend",
    }
    
    // Create pod
    pod := pm.CreatePod("web-pod", "default", containers, labels)
    
    // Schedule pod
    pm.SchedulePod("web-pod", "node-1")
    
    // Update status
    pm.UpdatePodStatus("web-pod", "Running")
    
    // Get pod
    retrievedPod := pm.GetPod("web-pod")
    if retrievedPod != nil {
        fmt.Printf("Retrieved pod: %s (status: %s)\n", retrievedPod.Name, retrievedPod.Status)
    }
    
    // List pods
    pods := pm.ListPods("default")
    fmt.Printf("Pods in default namespace: %d\n", len(pods))
    
    // Get pod logs
    logs := pm.GetPodLogs("web-pod")
    fmt.Printf("Pod logs: %v\n", logs)
    
    // Get pod metrics
    metrics := pm.GetPodMetrics("web-pod")
    fmt.Printf("Pod metrics: %v\n", metrics)
    
    // Scale pod
    pm.ScalePod("web-pod", 3)
    
    // Get pod events
    events := pm.GetPodEvents("web-pod")
    fmt.Printf("Pod events: %v\n", events)
    
    // Delete pod
    pm.DeletePod("web-pod")
}
```

## Service Discovery

### Theory

Service discovery allows pods to find and communicate with each other. Kubernetes provides services that act as stable network endpoints for pods.

### Service Discovery Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "net"
    "sync"
    "time"
)

type Service struct {
    Name        string
    Namespace   string
    Type        string
    Port        int
    TargetPort  int
    Selector    map[string]string
    Endpoints   []Endpoint
    Status      string
    CreatedAt   time.Time
}

type Endpoint struct {
    IP   string
    Port int
    Name string
}

type ServiceDiscovery struct {
    Services map[string]*Service
    mutex    sync.RWMutex
}

func NewServiceDiscovery() *ServiceDiscovery {
    return &ServiceDiscovery{
        Services: make(map[string]*Service),
    }
}

func (sd *ServiceDiscovery) CreateService(name, namespace, serviceType string, port, targetPort int, selector map[string]string) *Service {
    service := &Service{
        Name:       name,
        Namespace:  namespace,
        Type:       serviceType,
        Port:       port,
        TargetPort: targetPort,
        Selector:   selector,
        Endpoints:  []Endpoint{},
        Status:     "Pending",
        CreatedAt:  time.Now(),
    }
    
    sd.mutex.Lock()
    sd.Services[name] = service
    sd.mutex.Unlock()
    
    fmt.Printf("Created service: %s in namespace %s\n", name, namespace)
    return service
}

func (sd *ServiceDiscovery) AddEndpoint(serviceName string, endpoint Endpoint) {
    sd.mutex.Lock()
    defer sd.mutex.Unlock()
    
    if service, exists := sd.Services[serviceName]; exists {
        service.Endpoints = append(service.Endpoints, endpoint)
        service.Status = "Ready"
        fmt.Printf("Added endpoint %s:%d to service %s\n", endpoint.IP, endpoint.Port, serviceName)
    }
}

func (sd *ServiceDiscovery) RemoveEndpoint(serviceName string, endpointIP string) {
    sd.mutex.Lock()
    defer sd.mutex.Unlock()
    
    if service, exists := sd.Services[serviceName]; exists {
        for i, ep := range service.Endpoints {
            if ep.IP == endpointIP {
                service.Endpoints = append(service.Endpoints[:i], service.Endpoints[i+1:]...)
                fmt.Printf("Removed endpoint %s from service %s\n", endpointIP, serviceName)
                break
            }
        }
    }
}

func (sd *ServiceDiscovery) GetService(name string) *Service {
    sd.mutex.RLock()
    defer sd.mutex.RUnlock()
    
    return sd.Services[name]
}

func (sd *ServiceDiscovery) ListServices(namespace string) []*Service {
    sd.mutex.RLock()
    defer sd.mutex.RUnlock()
    
    var services []*Service
    for _, service := range sd.Services {
        if namespace == "" || service.Namespace == namespace {
            services = append(services, service)
        }
    }
    
    return services
}

func (sd *ServiceDiscovery) ResolveService(serviceName string) ([]Endpoint, error) {
    sd.mutex.RLock()
    service, exists := sd.Services[serviceName]
    sd.mutex.RUnlock()
    
    if !exists {
        return nil, fmt.Errorf("service %s not found", serviceName)
    }
    
    if len(service.Endpoints) == 0 {
        return nil, fmt.Errorf("no endpoints available for service %s", serviceName)
    }
    
    return service.Endpoints, nil
}

func (sd *ServiceDiscovery) LoadBalance(serviceName string) (Endpoint, error) {
    endpoints, err := sd.ResolveService(serviceName)
    if err != nil {
        return Endpoint{}, err
    }
    
    // Simple round-robin load balancing
    index := time.Now().UnixNano() % int64(len(endpoints))
    return endpoints[index], nil
}

func (sd *ServiceDiscovery) HealthCheck(serviceName string) bool {
    sd.mutex.RLock()
    service, exists := sd.Services[serviceName]
    sd.mutex.RUnlock()
    
    if !exists {
        return false
    }
    
    if len(service.Endpoints) == 0 {
        return false
    }
    
    // Check if at least one endpoint is healthy
    for _, endpoint := range service.Endpoints {
        if sd.isEndpointHealthy(endpoint) {
            return true
        }
    }
    
    return false
}

func (sd *ServiceDiscovery) isEndpointHealthy(endpoint Endpoint) bool {
    // Simulate health check
    conn, err := net.DialTimeout("tcp", fmt.Sprintf("%s:%d", endpoint.IP, endpoint.Port), 1*time.Second)
    if err != nil {
        return false
    }
    conn.Close()
    return true
}

func (sd *ServiceDiscovery) GetServiceDNS(serviceName, namespace string) string {
    if namespace == "" {
        namespace = "default"
    }
    return fmt.Sprintf("%s.%s.svc.cluster.local", serviceName, namespace)
}

func (sd *ServiceDiscovery) UpdateServiceStatus(serviceName, status string) {
    sd.mutex.Lock()
    defer sd.mutex.Unlock()
    
    if service, exists := sd.Services[serviceName]; exists {
        service.Status = status
        fmt.Printf("Updated service %s status to: %s\n", serviceName, status)
    }
}

func (sd *ServiceDiscovery) DeleteService(serviceName string) bool {
    sd.mutex.Lock()
    defer sd.mutex.Unlock()
    
    if _, exists := sd.Services[serviceName]; exists {
        delete(sd.Services, serviceName)
        fmt.Printf("Deleted service: %s\n", serviceName)
        return true
    }
    
    return false
}

func main() {
    sd := NewServiceDiscovery()
    
    fmt.Println("Kubernetes Service Discovery Demo:")
    
    // Create service
    selector := map[string]string{
        "app":     "web",
        "version": "1.0",
    }
    
    service := sd.CreateService("web-service", "default", "ClusterIP", 80, 8080, selector)
    
    // Add endpoints
    sd.AddEndpoint("web-service", Endpoint{IP: "10.0.1.1", Port: 8080, Name: "web-pod-1"})
    sd.AddEndpoint("web-service", Endpoint{IP: "10.0.1.2", Port: 8080, Name: "web-pod-2"})
    sd.AddEndpoint("web-service", Endpoint{IP: "10.0.1.3", Port: 8080, Name: "web-pod-3"})
    
    // Resolve service
    endpoints, err := sd.ResolveService("web-service")
    if err != nil {
        fmt.Printf("Error resolving service: %v\n", err)
    } else {
        fmt.Printf("Service endpoints: %v\n", endpoints)
    }
    
    // Load balance
    endpoint, err := sd.LoadBalance("web-service")
    if err != nil {
        fmt.Printf("Error load balancing: %v\n", err)
    } else {
        fmt.Printf("Selected endpoint: %v\n", endpoint)
    }
    
    // Health check
    healthy := sd.HealthCheck("web-service")
    fmt.Printf("Service is healthy: %v\n", healthy)
    
    // Get DNS name
    dns := sd.GetServiceDNS("web-service", "default")
    fmt.Printf("Service DNS: %s\n", dns)
    
    // Update status
    sd.UpdateServiceStatus("web-service", "Ready")
    
    // List services
    services := sd.ListServices("default")
    fmt.Printf("Services in default namespace: %d\n", len(services))
    
    // Remove endpoint
    sd.RemoveEndpoint("web-service", "10.0.1.1")
    
    // Delete service
    sd.DeleteService("web-service")
}
```

## Deployment Strategies

### Theory

Deployment strategies define how applications are updated in Kubernetes. Common strategies include rolling updates, blue-green deployments, and canary deployments.

### Deployment Strategies Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Deployment struct {
    Name        string
    Namespace   string
    Replicas    int
    Strategy    string
    Image       string
    Version     string
    Status      string
    CreatedAt   time.Time
    UpdatedAt   time.Time
}

type DeploymentStrategy struct {
    Deployments map[string]*Deployment
    mutex       sync.RWMutex
}

func NewDeploymentStrategy() *DeploymentStrategy {
    return &DeploymentStrategy{
        Deployments: make(map[string]*Deployment),
    }
}

func (ds *DeploymentStrategy) CreateDeployment(name, namespace string, replicas int, strategy, image, version string) *Deployment {
    deployment := &Deployment{
        Name:      name,
        Namespace: namespace,
        Replicas:  replicas,
        Strategy:  strategy,
        Image:     image,
        Version:   version,
        Status:    "Pending",
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
    
    ds.mutex.Lock()
    ds.Deployments[name] = deployment
    ds.mutex.Unlock()
    
    fmt.Printf("Created deployment: %s with strategy %s\n", name, strategy)
    return deployment
}

func (ds *DeploymentStrategy) RollingUpdate(deploymentName, newImage, newVersion string) error {
    ds.mutex.Lock()
    deployment, exists := ds.Deployments[deploymentName]
    ds.mutex.Unlock()
    
    if !exists {
        return fmt.Errorf("deployment %s not found", deploymentName)
    }
    
    fmt.Printf("Starting rolling update for deployment %s\n", deploymentName)
    
    // Simulate rolling update
    for i := 0; i < deployment.Replicas; i++ {
        fmt.Printf("Updating replica %d/%d\n", i+1, deployment.Replicas)
        time.Sleep(100 * time.Millisecond) // Simulate update time
    }
    
    deployment.Image = newImage
    deployment.Version = newVersion
    deployment.Status = "Ready"
    deployment.UpdatedAt = time.Now()
    
    fmt.Printf("Rolling update completed for deployment %s\n", deploymentName)
    return nil
}

func (ds *DeploymentStrategy) BlueGreenDeployment(deploymentName, newImage, newVersion string) error {
    ds.mutex.Lock()
    deployment, exists := ds.Deployments[deploymentName]
    ds.mutex.Unlock()
    
    if !exists {
        return fmt.Errorf("deployment %s not found", deploymentName)
    }
    
    fmt.Printf("Starting blue-green deployment for %s\n", deploymentName)
    
    // Create green environment
    greenDeployment := &Deployment{
        Name:      deploymentName + "-green",
        Namespace: deployment.Namespace,
        Replicas:  deployment.Replicas,
        Strategy:  "blue-green",
        Image:     newImage,
        Version:   newVersion,
        Status:    "Deploying",
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
    
    ds.mutex.Lock()
    ds.Deployments[greenDeployment.Name] = greenDeployment
    ds.mutex.Unlock()
    
    // Deploy green environment
    fmt.Printf("Deploying green environment for %s\n", deploymentName)
    time.Sleep(500 * time.Millisecond) // Simulate deployment time
    
    // Switch traffic to green
    fmt.Printf("Switching traffic to green environment for %s\n", deploymentName)
    time.Sleep(100 * time.Millisecond)
    
    // Update original deployment
    deployment.Image = newImage
    deployment.Version = newVersion
    deployment.Status = "Ready"
    deployment.UpdatedAt = time.Now()
    
    // Clean up green deployment
    ds.mutex.Lock()
    delete(ds.Deployments, greenDeployment.Name)
    ds.mutex.Unlock()
    
    fmt.Printf("Blue-green deployment completed for %s\n", deploymentName)
    return nil
}

func (ds *DeploymentStrategy) CanaryDeployment(deploymentName, newImage, newVersion string, canaryPercentage int) error {
    ds.mutex.Lock()
    deployment, exists := ds.Deployments[deploymentName]
    ds.mutex.Unlock()
    
    if !exists {
        return fmt.Errorf("deployment %s not found", deploymentName)
    }
    
    fmt.Printf("Starting canary deployment for %s with %d%% traffic\n", deploymentName, canaryPercentage)
    
    // Create canary deployment
    canaryDeployment := &Deployment{
        Name:      deploymentName + "-canary",
        Namespace: deployment.Namespace,
        Replicas:  (deployment.Replicas * canaryPercentage) / 100,
        Strategy:  "canary",
        Image:     newImage,
        Version:   newVersion,
        Status:    "Deploying",
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
    
    ds.mutex.Lock()
    ds.Deployments[canaryDeployment.Name] = canaryDeployment
    ds.mutex.Unlock()
    
    // Deploy canary
    fmt.Printf("Deploying canary version for %s\n", deploymentName)
    time.Sleep(300 * time.Millisecond) // Simulate deployment time
    
    // Monitor canary
    fmt.Printf("Monitoring canary deployment for %s\n", deploymentName)
    time.Sleep(200 * time.Millisecond)
    
    // Promote canary to full deployment
    fmt.Printf("Promoting canary to full deployment for %s\n", deploymentName)
    deployment.Image = newImage
    deployment.Version = newVersion
    deployment.Status = "Ready"
    deployment.UpdatedAt = time.Now()
    
    // Clean up canary deployment
    ds.mutex.Lock()
    delete(ds.Deployments, canaryDeployment.Name)
    ds.mutex.Unlock()
    
    fmt.Printf("Canary deployment completed for %s\n", deploymentName)
    return nil
}

func (ds *DeploymentStrategy) Rollback(deploymentName string) error {
    ds.mutex.Lock()
    deployment, exists := ds.Deployments[deploymentName]
    ds.mutex.Unlock()
    
    if !exists {
        return fmt.Errorf("deployment %s not found", deploymentName)
    }
    
    fmt.Printf("Rolling back deployment %s\n", deploymentName)
    
    // Simulate rollback
    time.Sleep(200 * time.Millisecond)
    
    // Update deployment status
    deployment.Status = "RolledBack"
    deployment.UpdatedAt = time.Now()
    
    fmt.Printf("Rollback completed for deployment %s\n", deploymentName)
    return nil
}

func (ds *DeploymentStrategy) ScaleDeployment(deploymentName string, replicas int) error {
    ds.mutex.Lock()
    deployment, exists := ds.Deployments[deploymentName]
    ds.mutex.Unlock()
    
    if !exists {
        return fmt.Errorf("deployment %s not found", deploymentName)
    }
    
    fmt.Printf("Scaling deployment %s to %d replicas\n", deploymentName, replicas)
    
    // Simulate scaling
    time.Sleep(100 * time.Millisecond)
    
    deployment.Replicas = replicas
    deployment.UpdatedAt = time.Now()
    
    fmt.Printf("Scaling completed for deployment %s\n", deploymentName)
    return nil
}

func (ds *DeploymentStrategy) GetDeployment(name string) *Deployment {
    ds.mutex.RLock()
    defer ds.mutex.RUnlock()
    
    return ds.Deployments[name]
}

func (ds *DeploymentStrategy) ListDeployments(namespace string) []*Deployment {
    ds.mutex.RLock()
    defer ds.mutex.RUnlock()
    
    var deployments []*Deployment
    for _, deployment := range ds.Deployments {
        if namespace == "" || deployment.Namespace == namespace {
            deployments = append(deployments, deployment)
        }
    }
    
    return deployments
}

func (ds *DeploymentStrategy) GetDeploymentStatus(name string) map[string]interface{} {
    ds.mutex.RLock()
    deployment, exists := ds.Deployments[name]
    ds.mutex.RUnlock()
    
    if !exists {
        return map[string]interface{}{}
    }
    
    return map[string]interface{}{
        "name":      deployment.Name,
        "namespace": deployment.Namespace,
        "replicas":  deployment.Replicas,
        "strategy":  deployment.Strategy,
        "image":     deployment.Image,
        "version":   deployment.Version,
        "status":    deployment.Status,
        "uptime":    time.Since(deployment.CreatedAt).String(),
    }
}

func main() {
    ds := NewDeploymentStrategy()
    
    fmt.Println("Kubernetes Deployment Strategies Demo:")
    
    // Create deployment
    deployment := ds.CreateDeployment("web-app", "default", 3, "rolling", "nginx:1.20", "v1.0")
    
    // Rolling update
    err := ds.RollingUpdate("web-app", "nginx:1.21", "v1.1")
    if err != nil {
        fmt.Printf("Error in rolling update: %v\n", err)
    }
    
    // Blue-green deployment
    err = ds.BlueGreenDeployment("web-app", "nginx:1.22", "v1.2")
    if err != nil {
        fmt.Printf("Error in blue-green deployment: %v\n", err)
    }
    
    // Canary deployment
    err = ds.CanaryDeployment("web-app", "nginx:1.23", "v1.3", 20)
    if err != nil {
        fmt.Printf("Error in canary deployment: %v\n", err)
    }
    
    // Scale deployment
    err = ds.ScaleDeployment("web-app", 5)
    if err != nil {
        fmt.Printf("Error scaling deployment: %v\n", err)
    }
    
    // Get deployment status
    status := ds.GetDeploymentStatus("web-app")
    fmt.Printf("Deployment status: %v\n", status)
    
    // List deployments
    deployments := ds.ListDeployments("default")
    fmt.Printf("Deployments in default namespace: %d\n", len(deployments))
    
    // Rollback
    err = ds.Rollback("web-app")
    if err != nil {
        fmt.Printf("Error rolling back: %v\n", err)
    }
}
```

## Follow-up Questions

### 1. Pod Management
**Q: What is the difference between a pod and a container?**
A: A pod is a Kubernetes abstraction that can contain one or more containers. Containers within a pod share the same network and storage resources and are always scheduled together.

### 2. Service Discovery
**Q: What are the different types of Kubernetes services?**
A: ClusterIP (internal), NodePort (expose on node), LoadBalancer (cloud load balancer), and ExternalName (external service).

### 3. Deployment Strategies
**Q: When would you use blue-green deployment vs canary deployment?**
A: Use blue-green for immediate, complete switches with rollback capability. Use canary for gradual rollouts with monitoring and risk mitigation.

## Complexity Analysis

| Operation | Pod Management | Service Discovery | Deployment Strategies |
|-----------|----------------|-------------------|---------------------|
| Create | O(1) | O(1) | O(1) |
| Update | O(1) | O(1) | O(n) |
| Delete | O(1) | O(1) | O(1) |
| List | O(n) | O(n) | O(n) |
| Scale | O(1) | N/A | O(1) |

## Applications

1. **Pod Management**: Container orchestration, microservices deployment, resource management
2. **Service Discovery**: Load balancing, service mesh, microservices communication
3. **Deployment Strategies**: CI/CD pipelines, zero-downtime deployments, risk management
4. **Kubernetes Orchestration**: Cloud-native applications, container management, scalability

---

**Next**: [Container Orchestration](container-orchestration.md) | **Previous**: [Cloud Architecture](README.md) | **Up**: [Cloud Architecture](README.md)
