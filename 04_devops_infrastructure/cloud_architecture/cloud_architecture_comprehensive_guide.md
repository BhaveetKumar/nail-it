# ☁️ Cloud Architecture & Services Comprehensive Guide

## Table of Contents
1. [Cloud Computing Models](#cloud-computing-models/)
2. [AWS Services](#aws-services/)
3. [Google Cloud Platform](#google-cloud-platform/)
4. [Azure Services](#azure-services/)
5. [Container Orchestration](#container-orchestration/)
6. [Serverless Architecture](#serverless-architecture/)
7. [Cloud Security](#cloud-security/)
8. [Cost Optimization](#cost-optimization/)
9. [Go Implementation Examples](#go-implementation-examples/)
10. [Interview Questions](#interview-questions/)

## Cloud Computing Models

### Infrastructure as a Service (IaaS)

```go
package main

import (
    "fmt"
    "time"
)

// Virtual Machine Management
type VirtualMachine struct {
    ID          string
    Name        string
    InstanceType string
    State       VMState
    Region      string
    Zone        string
    CreatedAt   time.Time
    Tags        map[string]string
}

type VMState int

const (
    Pending VMState = iota
    Running
    Stopped
    Terminated
)

type VMManager struct {
    instances map[string]*VirtualMachine
}

func NewVMManager() *VMManager {
    return &VMManager{
        instances: make(map[string]*VirtualMachine),
    }
}

func (vm *VMManager) CreateInstance(name, instanceType, region, zone string) (*VirtualMachine, error) {
    vmID := fmt.Sprintf("i-%d", time.Now().UnixNano())
    
    instance := &VirtualMachine{
        ID:           vmID,
        Name:         name,
        InstanceType: instanceType,
        State:        Pending,
        Region:       region,
        Zone:         zone,
        CreatedAt:    time.Now(),
        Tags:         make(map[string]string),
    }
    
    vm.instances[vmID] = instance
    
    // Simulate instance startup
    go func() {
        time.Sleep(2 * time.Second)
        instance.State = Running
        fmt.Printf("Instance %s is now running\n", vmID)
    }()
    
    return instance, nil
}

func (vm *VMManager) StartInstance(vmID string) error {
    instance, exists := vm.instances[vmID]
    if !exists {
        return fmt.Errorf("instance not found")
    }
    
    if instance.State == Running {
        return fmt.Errorf("instance already running")
    }
    
    instance.State = Running
    return nil
}

func (vm *VMManager) StopInstance(vmID string) error {
    instance, exists := vm.instances[vmID]
    if !exists {
        return fmt.Errorf("instance not found")
    }
    
    instance.State = Stopped
    return nil
}

func (vm *VMManager) TerminateInstance(vmID string) error {
    instance, exists := vm.instances[vmID]
    if !exists {
        return fmt.Errorf("instance not found")
    }
    
    instance.State = Terminated
    delete(vm.instances, vmID)
    return nil
}

// Auto Scaling Group
type AutoScalingGroup struct {
    ID                string
    Name              string
    MinSize           int
    MaxSize           int
    DesiredCapacity   int
    InstanceType      string
    LaunchTemplate    string
    TargetGroupARNs   []string
    HealthCheckType   string
    Instances         []*VirtualMachine
}

type ScalingPolicy struct {
    Name              string
    AdjustmentType    string
    ScalingAdjustment int
    Cooldown          time.Duration
    MetricName        string
    Threshold         float64
}

func (asg *AutoScalingGroup) ScaleOut() error {
    if asg.DesiredCapacity >= asg.MaxSize {
        return fmt.Errorf("cannot scale beyond max size")
    }
    
    asg.DesiredCapacity++
    
    // Create new instance
    vm := &VirtualMachine{
        ID:           fmt.Sprintf("i-%d", time.Now().UnixNano()),
        Name:         fmt.Sprintf("%s-instance-%d", asg.Name, len(asg.Instances)+1),
        InstanceType: asg.InstanceType,
        State:        Pending,
        CreatedAt:    time.Now(),
        Tags:         make(map[string]string),
    }
    
    asg.Instances = append(asg.Instances, vm)
    
    // Simulate instance startup
    go func() {
        time.Sleep(2 * time.Second)
        vm.State = Running
        fmt.Printf("Scaled out: %s\n", vm.Name)
    }()
    
    return nil
}

func (asg *AutoScalingGroup) ScaleIn() error {
    if asg.DesiredCapacity <= asg.MinSize {
        return fmt.Errorf("cannot scale below min size")
    }
    
    if len(asg.Instances) == 0 {
        return fmt.Errorf("no instances to terminate")
    }
    
    // Remove last instance
    instance := asg.Instances[len(asg.Instances)-1]
    asg.Instances = asg.Instances[:len(asg.Instances)-1]
    asg.DesiredCapacity--
    
    instance.State = Terminated
    fmt.Printf("Scaled in: %s\n", instance.Name)
    
    return nil
}
```

### Platform as a Service (PaaS)

```go
package main

import (
    "fmt"
    "time"
)

// Application Platform
type Application struct {
    ID          string
    Name        string
    Runtime     string
    Version     string
    Status      AppStatus
    Instances   int
    Memory      int
    CPU         int
    Environment map[string]string
    CreatedAt   time.Time
}

type AppStatus int

const (
    Deploying AppStatus = iota
    Running
    Stopped
    Failed
)

type PaaSPlatform struct {
    applications map[string]*Application
    runtimes     []string
}

func NewPaaSPlatform() *PaaSPlatform {
    return &PaaSPlatform{
        applications: make(map[string]*Application),
        runtimes:     []string{"go", "nodejs", "python", "java"},
    }
}

func (p *PaaSPlatform) DeployApplication(name, runtime, version string, instances, memory, cpu int) (*Application, error) {
    appID := fmt.Sprintf("app-%d", time.Now().UnixNano())
    
    app := &Application{
        ID:          appID,
        Name:        name,
        Runtime:     runtime,
        Version:     version,
        Status:      Deploying,
        Instances:   instances,
        Memory:      memory,
        CPU:         cpu,
        Environment: make(map[string]string),
        CreatedAt:   time.Now(),
    }
    
    p.applications[appID] = app
    
    // Simulate deployment
    go func() {
        time.Sleep(3 * time.Second)
        app.Status = Running
        fmt.Printf("Application %s deployed successfully\n", name)
    }()
    
    return app, nil
}

func (p *PaaSPlatform) ScaleApplication(appID string, instances int) error {
    app, exists := p.applications[appID]
    if !exists {
        return fmt.Errorf("application not found")
    }
    
    app.Instances = instances
    fmt.Printf("Scaled application %s to %d instances\n", app.Name, instances)
    
    return nil
}

func (p *PaaSPlatform) UpdateEnvironment(appID string, envVars map[string]string) error {
    app, exists := p.applications[appID]
    if !exists {
        return fmt.Errorf("application not found")
    }
    
    for key, value := range envVars {
        app.Environment[key] = value
    }
    
    fmt.Printf("Updated environment for application %s\n", app.Name)
    return nil
}

func (p *PaaSPlatform) GetApplicationStatus(appID string) (*Application, error) {
    app, exists := p.applications[appID]
    if !exists {
        return nil, fmt.Errorf("application not found")
    }
    
    return app, nil
}
```

## AWS Services

### EC2 and Auto Scaling

```go
package main

import (
    "fmt"
    "time"
)

// EC2 Instance Management
type EC2Instance struct {
    InstanceID     string
    InstanceType   string
    State          string
    PublicIP       string
    PrivateIP      string
    SecurityGroups []string
    KeyName        string
    LaunchTime     time.Time
}

type EC2Manager struct {
    instances map[string]*EC2Instance
    regions   []string
}

func NewEC2Manager() *EC2Manager {
    return &EC2Manager{
        instances: make(map[string]*EC2Instance),
        regions:   []string{"us-east-1", "us-west-2", "eu-west-1"},
    }
}

func (ec2 *EC2Manager) RunInstances(instanceType, keyName string, securityGroups []string) (*EC2Instance, error) {
    instanceID := fmt.Sprintf("i-%d", time.Now().UnixNano())
    
    instance := &EC2Instance{
        InstanceID:     instanceID,
        InstanceType:   instanceType,
        State:          "pending",
        SecurityGroups: securityGroups,
        KeyName:        keyName,
        LaunchTime:     time.Now(),
    }
    
    ec2.instances[instanceID] = instance
    
    // Simulate instance startup
    go func() {
        time.Sleep(2 * time.Second)
        instance.State = "running"
        instance.PublicIP = fmt.Sprintf("54.%d.%d.%d", 
            time.Now().Unix()%255, 
            time.Now().Unix()%255, 
            time.Now().Unix()%255)
        instance.PrivateIP = fmt.Sprintf("10.%d.%d.%d", 
            time.Now().Unix()%255, 
            time.Now().Unix()%255, 
            time.Now().Unix()%255)
        fmt.Printf("Instance %s is now running\n", instanceID)
    }()
    
    return instance, nil
}

func (ec2 *EC2Manager) TerminateInstance(instanceID string) error {
    instance, exists := ec2.instances[instanceID]
    if !exists {
        return fmt.Errorf("instance not found")
    }
    
    instance.State = "terminated"
    delete(ec2.instances, instanceID)
    fmt.Printf("Instance %s terminated\n", instanceID)
    
    return nil
}

// S3 Bucket Management
type S3Bucket struct {
    Name         string
    Region       string
    CreationDate time.Time
    Objects      map[string]*S3Object
}

type S3Object struct {
    Key          string
    Size         int64
    LastModified time.Time
    ETag         string
    ContentType  string
}

type S3Manager struct {
    buckets map[string]*S3Bucket
}

func NewS3Manager() *S3Manager {
    return &S3Manager{
        buckets: make(map[string]*S3Bucket),
    }
}

func (s3 *S3Manager) CreateBucket(name, region string) (*S3Bucket, error) {
    bucket := &S3Bucket{
        Name:         name,
        Region:       region,
        CreationDate: time.Now(),
        Objects:      make(map[string]*S3Object),
    }
    
    s3.buckets[name] = bucket
    fmt.Printf("Bucket %s created in region %s\n", name, region)
    
    return bucket, nil
}

func (s3 *S3Manager) PutObject(bucketName, key string, data []byte, contentType string) error {
    bucket, exists := s3.buckets[bucketName]
    if !exists {
        return fmt.Errorf("bucket not found")
    }
    
    object := &S3Object{
        Key:          key,
        Size:         int64(len(data)),
        LastModified: time.Now(),
        ETag:         fmt.Sprintf("%x", data),
        ContentType:  contentType,
    }
    
    bucket.Objects[key] = object
    fmt.Printf("Object %s uploaded to bucket %s\n", key, bucketName)
    
    return nil
}

func (s3 *S3Manager) GetObject(bucketName, key string) (*S3Object, error) {
    bucket, exists := s3.buckets[bucketName]
    if !exists {
        return nil, fmt.Errorf("bucket not found")
    }
    
    object, exists := bucket.Objects[key]
    if !exists {
        return nil, fmt.Errorf("object not found")
    }
    
    return object, nil
}

// RDS Database Management
type RDSInstance struct {
    DBInstanceIdentifier string
    Engine               string
    EngineVersion        string
    DBInstanceClass      string
    AllocatedStorage     int
    Status               string
    Endpoint             string
    Port                 int
    MasterUsername       string
    CreatedAt            time.Time
}

type RDSManager struct {
    instances map[string]*RDSInstance
}

func NewRDSManager() *RDSManager {
    return &RDSManager{
        instances: make(map[string]*RDSInstance),
    }
}

func (rds *RDSManager) CreateDBInstance(identifier, engine, version, instanceClass string, allocatedStorage int) (*RDSInstance, error) {
    instance := &RDSInstance{
        DBInstanceIdentifier: identifier,
        Engine:               engine,
        EngineVersion:        version,
        DBInstanceClass:      instanceClass,
        AllocatedStorage:     allocatedStorage,
        Status:               "creating",
        Endpoint:             fmt.Sprintf("%s.region.rds.amazonaws.com", identifier),
        Port:                 3306,
        MasterUsername:       "admin",
        CreatedAt:            time.Now(),
    }
    
    rds.instances[identifier] = instance
    
    // Simulate database creation
    go func() {
        time.Sleep(5 * time.Second)
        instance.Status = "available"
        fmt.Printf("RDS instance %s is now available\n", identifier)
    }()
    
    return instance, nil
}

func (rds *RDSManager) DeleteDBInstance(identifier string) error {
    instance, exists := rds.instances[identifier]
    if !exists {
        return fmt.Errorf("instance not found")
    }
    
    instance.Status = "deleting"
    delete(rds.instances, identifier)
    fmt.Printf("RDS instance %s deleted\n", identifier)
    
    return nil
}
```

## Google Cloud Platform

### Compute Engine and Cloud Functions

```go
package main

import (
    "fmt"
    "time"
)

// Compute Engine Instance
type GCEInstance struct {
    Name         string
    MachineType  string
    Zone         string
    Status       string
    ExternalIP   string
    InternalIP   string
    CreatedAt    time.Time
    Labels       map[string]string
}

type GCEManager struct {
    instances map[string]*GCEInstance
    zones     []string
}

func NewGCEManager() *GCEManager {
    return &GCEManager{
        instances: make(map[string]*GCEInstance),
        zones:     []string{"us-central1-a", "us-central1-b", "europe-west1-a"},
    }
}

func (gce *GCEManager) CreateInstance(name, machineType, zone string) (*GCEInstance, error) {
    instance := &GCEInstance{
        Name:        name,
        MachineType: machineType,
        Zone:        zone,
        Status:      "PROVISIONING",
        CreatedAt:   time.Now(),
        Labels:      make(map[string]string),
    }
    
    gce.instances[name] = instance
    
    // Simulate instance creation
    go func() {
        time.Sleep(3 * time.Second)
        instance.Status = "RUNNING"
        instance.ExternalIP = fmt.Sprintf("35.%d.%d.%d", 
            time.Now().Unix()%255, 
            time.Now().Unix()%255, 
            time.Now().Unix()%255)
        instance.InternalIP = fmt.Sprintf("10.%d.%d.%d", 
            time.Now().Unix()%255, 
            time.Now().Unix()%255, 
            time.Now().Unix()%255)
        fmt.Printf("GCE instance %s is now running\n", name)
    }()
    
    return instance, nil
}

// Cloud Functions
type CloudFunction struct {
    Name        string
    Runtime     string
    Trigger     string
    Status      string
    Memory      int
    Timeout     int
    CreatedAt   time.Time
    Invocations int
}

type CloudFunctionsManager struct {
    functions map[string]*CloudFunction
}

func NewCloudFunctionsManager() *CloudFunctionsManager {
    return &CloudFunctionsManager{
        functions: make(map[string]*CloudFunction),
    }
}

func (cf *CloudFunctionsManager) DeployFunction(name, runtime, trigger string, memory, timeout int) (*CloudFunction, error) {
    function := &CloudFunction{
        Name:      name,
        Runtime:   runtime,
        Trigger:   trigger,
        Status:    "DEPLOYING",
        Memory:    memory,
        Timeout:   timeout,
        CreatedAt: time.Now(),
    }
    
    cf.functions[name] = function
    
    // Simulate function deployment
    go func() {
        time.Sleep(2 * time.Second)
        function.Status = "ACTIVE"
        fmt.Printf("Cloud Function %s deployed successfully\n", name)
    }()
    
    return function, nil
}

func (cf *CloudFunctionsManager) InvokeFunction(name string, payload map[string]interface{}) (interface{}, error) {
    function, exists := cf.functions[name]
    if !exists {
        return nil, fmt.Errorf("function not found")
    }
    
    if function.Status != "ACTIVE" {
        return nil, fmt.Errorf("function not active")
    }
    
    function.Invocations++
    fmt.Printf("Function %s invoked with payload: %v\n", name, payload)
    
    // Simulate function execution
    return map[string]interface{}{
        "statusCode": 200,
        "body":       "Function executed successfully",
        "timestamp":  time.Now().Unix(),
    }, nil
}

// Cloud Storage
type CloudStorageBucket struct {
    Name         string
    Location     string
    StorageClass string
    CreatedAt    time.Time
    Objects      map[string]*StorageObject
}

type StorageObject struct {
    Name         string
    Size         int64
    ContentType  string
    Created      time.Time
    Updated      time.Time
    MD5Hash      string
}

type CloudStorageManager struct {
    buckets map[string]*CloudStorageBucket
}

func NewCloudStorageManager() *CloudStorageManager {
    return &CloudStorageManager{
        buckets: make(map[string]*CloudStorageBucket),
    }
}

func (cs *CloudStorageManager) CreateBucket(name, location, storageClass string) (*CloudStorageBucket, error) {
    bucket := &CloudStorageBucket{
        Name:         name,
        Location:     location,
        StorageClass: storageClass,
        CreatedAt:    time.Now(),
        Objects:      make(map[string]*StorageObject),
    }
    
    cs.buckets[name] = bucket
    fmt.Printf("Cloud Storage bucket %s created in %s\n", name, location)
    
    return bucket, nil
}

func (cs *CloudStorageManager) UploadObject(bucketName, objectName string, data []byte, contentType string) error {
    bucket, exists := cs.buckets[bucketName]
    if !exists {
        return fmt.Errorf("bucket not found")
    }
    
    object := &StorageObject{
        Name:        objectName,
        Size:        int64(len(data)),
        ContentType: contentType,
        Created:     time.Now(),
        Updated:     time.Now(),
        MD5Hash:     fmt.Sprintf("%x", data),
    }
    
    bucket.Objects[objectName] = object
    fmt.Printf("Object %s uploaded to bucket %s\n", objectName, bucketName)
    
    return nil
}
```

## Container Orchestration

### Kubernetes Implementation

```go
package main

import (
    "fmt"
    "time"
)

// Kubernetes Pod
type Pod struct {
    Name       string
    Namespace  string
    Status     string
    Node       string
    Containers []Container
    Labels     map[string]string
    CreatedAt  time.Time
}

type Container struct {
    Name  string
    Image string
    Port  int
    CPU   string
    Memory string
}

type KubernetesCluster struct {
    Name    string
    Nodes   map[string]*Node
    Pods    map[string]*Pod
    Services map[string]*Service
}

type Node struct {
    Name     string
    Status   string
    CPU      string
    Memory   string
    Pods     []string
    Labels   map[string]string
}

type Service struct {
    Name      string
    Namespace string
    Type      string
    Port      int
    TargetPort int
    Selector  map[string]string
    Endpoints []string
}

func NewKubernetesCluster(name string) *KubernetesCluster {
    return &KubernetesCluster{
        Name:     name,
        Nodes:    make(map[string]*Node),
        Pods:     make(map[string]*Pod),
        Services: make(map[string]*Service),
    }
}

func (k8s *KubernetesCluster) AddNode(name string, cpu, memory string) *Node {
    node := &Node{
        Name:   name,
        Status: "Ready",
        CPU:    cpu,
        Memory: memory,
        Pods:   make([]string, 0),
        Labels: make(map[string]string),
    }
    
    k8s.Nodes[name] = node
    fmt.Printf("Node %s added to cluster\n", name)
    
    return node
}

func (k8s *KubernetesCluster) CreatePod(name, namespace string, containers []Container, labels map[string]string) (*Pod, error) {
    // Find available node
    var targetNode string
    for nodeName, node := range k8s.Nodes {
        if node.Status == "Ready" {
            targetNode = nodeName
            break
        }
    }
    
    if targetNode == "" {
        return nil, fmt.Errorf("no available nodes")
    }
    
    pod := &Pod{
        Name:       name,
        Namespace:  namespace,
        Status:     "Pending",
        Node:       targetNode,
        Containers: containers,
        Labels:     labels,
        CreatedAt:  time.Now(),
    }
    
    k8s.Pods[name] = pod
    k8s.Nodes[targetNode].Pods = append(k8s.Nodes[targetNode].Pods, name)
    
    // Simulate pod scheduling
    go func() {
        time.Sleep(1 * time.Second)
        pod.Status = "Running"
        fmt.Printf("Pod %s is now running on node %s\n", name, targetNode)
    }()
    
    return pod, nil
}

func (k8s *KubernetesCluster) CreateService(name, namespace, serviceType string, port, targetPort int, selector map[string]string) (*Service, error) {
    service := &Service{
        Name:      name,
        Namespace: namespace,
        Type:      serviceType,
        Port:      port,
        TargetPort: targetPort,
        Selector:  selector,
        Endpoints: make([]string, 0),
    }
    
    // Find pods matching selector
    for podName, pod := range k8s.Pods {
        if pod.Namespace == namespace {
            match := true
            for key, value := range selector {
                if pod.Labels[key] != value {
                    match = false
                    break
                }
            }
            if match {
                service.Endpoints = append(service.Endpoints, podName)
            }
        }
    }
    
    k8s.Services[name] = service
    fmt.Printf("Service %s created with %d endpoints\n", name, len(service.Endpoints))
    
    return service, nil
}

func (k8s *KubernetesCluster) ScaleDeployment(deploymentName string, replicas int) error {
    // Find pods with deployment label
    var pods []string
    for podName, pod := range k8s.Pods {
        if pod.Labels["app"] == deploymentName {
            pods = append(pods, podName)
        }
    }
    
    currentReplicas := len(pods)
    
    if replicas > currentReplicas {
        // Scale up
        for i := 0; i < replicas-currentReplicas; i++ {
            podName := fmt.Sprintf("%s-%d", deploymentName, i)
            containers := []Container{
                {Name: "app", Image: "nginx:latest", Port: 80},
            }
            labels := map[string]string{"app": deploymentName}
            
            _, err := k8s.CreatePod(podName, "default", containers, labels)
            if err != nil {
                return err
            }
        }
    } else if replicas < currentReplicas {
        // Scale down
        for i := 0; i < currentReplicas-replicas; i++ {
            podName := pods[i]
            delete(k8s.Pods, podName)
            fmt.Printf("Pod %s terminated\n", podName)
        }
    }
    
    fmt.Printf("Scaled deployment %s to %d replicas\n", deploymentName, replicas)
    return nil
}
```

## Serverless Architecture

### AWS Lambda Implementation

```go
package main

import (
    "encoding/json"
    "fmt"
    "time"
)

// Lambda Function
type LambdaFunction struct {
    FunctionName string
    Runtime      string
    Handler      string
    Code         []byte
    MemorySize   int
    Timeout      int
    Environment  map[string]string
    CreatedAt    time.Time
    LastModified time.Time
}

type LambdaEvent struct {
    Source      string
    Detail      map[string]interface{}
    Time        time.Time
    ID          string
}

type LambdaContext struct {
    FunctionName    string
    FunctionVersion string
    InvokedFunctionARN string
    MemoryLimitInMB int
    RemainingTimeInMillis int
    RequestID       string
}

type LambdaManager struct {
    functions map[string]*LambdaFunction
}

func NewLambdaManager() *LambdaManager {
    return &LambdaManager{
        functions: make(map[string]*LambdaFunction),
    }
}

func (lm *LambdaManager) CreateFunction(name, runtime, handler string, code []byte, memorySize, timeout int) (*LambdaFunction, error) {
    function := &LambdaFunction{
        FunctionName: name,
        Runtime:      runtime,
        Handler:      handler,
        Code:         code,
        MemorySize:   memorySize,
        Timeout:      timeout,
        Environment:  make(map[string]string),
        CreatedAt:    time.Now(),
        LastModified: time.Now(),
    }
    
    lm.functions[name] = function
    fmt.Printf("Lambda function %s created\n", name)
    
    return function, nil
}

func (lm *LambdaManager) InvokeFunction(functionName string, event LambdaEvent) (interface{}, error) {
    function, exists := lm.functions[functionName]
    if !exists {
        return nil, fmt.Errorf("function not found")
    }
    
    context := LambdaContext{
        FunctionName:    function.FunctionName,
        FunctionVersion: "1",
        InvokedFunctionARN: fmt.Sprintf("arn:aws:lambda:us-east-1:123456789012:function:%s", functionName),
        MemoryLimitInMB: function.MemorySize,
        RemainingTimeInMillis: function.Timeout * 1000,
        RequestID:       fmt.Sprintf("req-%d", time.Now().UnixNano()),
    }
    
    // Simulate function execution
    result := map[string]interface{}{
        "statusCode": 200,
        "body":       fmt.Sprintf("Function %s executed successfully", functionName),
        "event":      event,
        "context":    context,
        "timestamp":  time.Now().Unix(),
    }
    
    fmt.Printf("Lambda function %s invoked\n", functionName)
    
    return result, nil
}

func (lm *LambdaManager) UpdateFunctionCode(functionName string, code []byte) error {
    function, exists := lm.functions[functionName]
    if !exists {
        return fmt.Errorf("function not found")
    }
    
    function.Code = code
    function.LastModified = time.Now()
    
    fmt.Printf("Lambda function %s code updated\n", functionName)
    return nil
}

func (lm *LambdaManager) SetEnvironmentVariables(functionName string, envVars map[string]string) error {
    function, exists := lm.functions[functionName]
    if !exists {
        return fmt.Errorf("function not found")
    }
    
    for key, value := range envVars {
        function.Environment[key] = value
    }
    
    fmt.Printf("Environment variables updated for function %s\n", functionName)
    return nil
}

// API Gateway
type APIGateway struct {
    Name        string
    Description string
    Endpoints   map[string]*Endpoint
    CreatedAt   time.Time
}

type Endpoint struct {
    Path        string
    Method      string
    LambdaARN   string
    AuthType    string
    RateLimit   int
    Timeout     int
}

type APIGatewayManager struct {
    gateways map[string]*APIGateway
}

func NewAPIGatewayManager() *APIGatewayManager {
    return &APIGatewayManager{
        gateways: make(map[string]*APIGateway),
    }
}

func (agm *APIGatewayManager) CreateAPI(name, description string) (*APIGateway, error) {
    api := &APIGateway{
        Name:        name,
        Description: description,
        Endpoints:   make(map[string]*Endpoint),
        CreatedAt:   time.Now(),
    }
    
    agm.gateways[name] = api
    fmt.Printf("API Gateway %s created\n", name)
    
    return api, nil
}

func (agm *APIGatewayManager) CreateEndpoint(apiName, path, method, lambdaARN string) (*Endpoint, error) {
    api, exists := agm.gateways[apiName]
    if !exists {
        return nil, fmt.Errorf("API not found")
    }
    
    endpointKey := fmt.Sprintf("%s %s", method, path)
    endpoint := &Endpoint{
        Path:      path,
        Method:    method,
        LambdaARN: lambdaARN,
        AuthType:  "NONE",
        RateLimit: 1000,
        Timeout:   30,
    }
    
    api.Endpoints[endpointKey] = endpoint
    fmt.Printf("Endpoint %s %s created for API %s\n", method, path, apiName)
    
    return endpoint, nil
}

func (agm *APIGatewayManager) InvokeEndpoint(apiName, path, method string, payload map[string]interface{}) (interface{}, error) {
    api, exists := agm.gateways[apiName]
    if !exists {
        return nil, fmt.Errorf("API not found")
    }
    
    endpointKey := fmt.Sprintf("%s %s", method, path)
    endpoint, exists := api.Endpoints[endpointKey]
    if !exists {
        return nil, fmt.Errorf("endpoint not found")
    }
    
    // Simulate API Gateway invocation
    result := map[string]interface{}{
        "statusCode": 200,
        "body":       fmt.Sprintf("Endpoint %s %s invoked successfully", method, path),
        "payload":    payload,
        "lambdaARN":  endpoint.LambdaARN,
        "timestamp":  time.Now().Unix(),
    }
    
    fmt.Printf("API Gateway endpoint %s %s invoked\n", method, path)
    
    return result, nil
}
```

## Interview Questions

### Basic Concepts
1. **What are the different cloud computing models?**
2. **Explain the difference between IaaS, PaaS, and SaaS.**
3. **What is container orchestration?**
4. **How does serverless computing work?**
5. **What are the benefits of cloud computing?**

### Advanced Topics
1. **How would you design a multi-region cloud architecture?**
2. **Explain the difference between horizontal and vertical scaling.**
3. **How do you handle data consistency in distributed cloud systems?**
4. **What are the security considerations for cloud deployments?**
5. **How would you optimize cloud costs?**

### System Design
1. **Design a scalable web application on AWS.**
2. **How would you implement a microservices architecture on Kubernetes?**
3. **Design a serverless data processing pipeline.**
4. **How would you implement disaster recovery in the cloud?**
5. **Design a multi-cloud architecture for high availability.**

## Conclusion

Cloud architecture is essential for building scalable, reliable, and cost-effective systems. Key areas to master:

- **Cloud Models**: IaaS, PaaS, SaaS understanding
- **Cloud Providers**: AWS, GCP, Azure services
- **Containerization**: Docker, Kubernetes orchestration
- **Serverless**: Functions, event-driven architecture
- **Security**: Cloud security best practices
- **Cost Optimization**: Resource management and cost control

Understanding these concepts helps in:
- Designing cloud-native applications
- Choosing appropriate cloud services
- Implementing scalable architectures
- Managing cloud resources efficiently
- Preparing for technical interviews

This guide provides a comprehensive foundation for cloud architecture concepts and their practical implementation in Go.
