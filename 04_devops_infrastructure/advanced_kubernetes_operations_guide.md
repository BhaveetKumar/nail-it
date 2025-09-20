# ⚙️ **Advanced Kubernetes Operations Guide**

> Comprehensive guide to advanced Kubernetes patterns, operators, service mesh, and cloud-native infrastructure for senior backend engineers

---

## 📋 **Table of Contents**

1. [Custom Resource Definitions (CRDs)](#-custom-resource-definitions-crds)
2. [Kubernetes Operators](#-kubernetes-operators)
3. [Advanced Networking & Service Mesh](#-advanced-networking--service-mesh)
4. [Security Policies & RBAC](#-security-policies--rbac)
5. [Multi-Cluster Management](#-multi-cluster-management)
6. [GitOps & Continuous Deployment](#-gitops--continuous-deployment)
7. [Advanced Scheduling & Resource Management](#-advanced-scheduling--resource-management)
8. [Observability & Monitoring](#-observability--monitoring)
9. [Disaster Recovery & Backup](#-disaster-recovery--backup)
10. [Interview Questions & Scenarios](#-interview-questions--scenarios)

---

## 🔧 **Custom Resource Definitions (CRDs)**

### **Advanced CRD Implementation**

```go
package crd

import (
    "context"
    "fmt"
    "time"

    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/runtime"
    "k8s.io/apimachinery/pkg/runtime/schema"
    "sigs.k8s.io/controller-runtime/pkg/client"
    "sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

// PaymentProcessor CRD for fintech applications
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:subresource:scale:specpath=.spec.replicas,statuspath=.status.replicas
// +kubebuilder:printcolumn:name="Status",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Replicas",type="integer",JSONPath=".spec.replicas"
// +kubebuilder:printcolumn:name="Ready",type="integer",JSONPath=".status.readyReplicas"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
type PaymentProcessor struct {
    metav1.TypeMeta   `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`

    Spec   PaymentProcessorSpec   `json:"spec,omitempty"`
    Status PaymentProcessorStatus `json:"status,omitempty"`
}

type PaymentProcessorSpec struct {
    // Replicas defines the desired number of payment processor instances
    // +kubebuilder:validation:Minimum=1
    // +kubebuilder:validation:Maximum=100
    // +kubebuilder:default=3
    Replicas int32 `json:"replicas"`

    // ProcessorType defines the type of payment processor
    // +kubebuilder:validation:Enum=card;upi;netbanking;wallet
    ProcessorType string `json:"processorType"`

    // Configuration for the payment processor
    Config PaymentProcessorConfig `json:"config"`

    // Resources defines resource requirements
    Resources ResourceRequirements `json:"resources,omitempty"`

    // SecurityContext defines security settings
    SecurityContext *SecurityContext `json:"securityContext,omitempty"`

    // Autoscaling configuration
    Autoscaling *AutoscalingConfig `json:"autoscaling,omitempty"`

    // Circuit breaker configuration
    CircuitBreaker *CircuitBreakerConfig `json:"circuitBreaker,omitempty"`

    // Monitoring configuration
    Monitoring *MonitoringConfig `json:"monitoring,omitempty"`
}

type PaymentProcessorConfig struct {
    // BankingPartners defines the banking partners configuration
    BankingPartners []BankingPartnerConfig `json:"bankingPartners"`

    // Timeout settings
    // +kubebuilder:validation:Pattern=^([0-9]+(\.[0-9]+)?(ns|us|µs|ms|s|m|h))+$
    Timeout string `json:"timeout,omitempty"`

    // Retry configuration
    RetryConfig *RetryConfig `json:"retryConfig,omitempty"`

    // Rate limiting configuration
    RateLimit *RateLimitConfig `json:"rateLimit,omitempty"`

    // Encryption configuration
    Encryption *EncryptionConfig `json:"encryption,omitempty"`
}

type BankingPartnerConfig struct {
    Name     string            `json:"name"`
    Endpoint string            `json:"endpoint"`
    Weight   int32             `json:"weight,omitempty"`
    Headers  map[string]string `json:"headers,omitempty"`
    
    // TLS configuration
    TLS *TLSConfig `json:"tls,omitempty"`
    
    // Authentication configuration
    Auth *AuthConfig `json:"auth,omitempty"`
}

type PaymentProcessorStatus struct {
    // Phase represents the current phase of the PaymentProcessor
    // +kubebuilder:validation:Enum=Pending;Running;Failed;Succeeded
    Phase string `json:"phase,omitempty"`

    // Replicas is the actual number of replicas
    Replicas int32 `json:"replicas,omitempty"`

    // ReadyReplicas is the number of ready replicas
    ReadyReplicas int32 `json:"readyReplicas,omitempty"`

    // Conditions represent the latest available observations
    Conditions []metav1.Condition `json:"conditions,omitempty"`

    // LastUpdateTime is the last time the status was updated
    LastUpdateTime *metav1.Time `json:"lastUpdateTime,omitempty"`

    // Metrics contains operational metrics
    Metrics *ProcessorMetrics `json:"metrics,omitempty"`

    // BankingPartnerStatus contains status of banking partners
    BankingPartnerStatus []BankingPartnerStatus `json:"bankingPartnerStatus,omitempty"`
}

type ProcessorMetrics struct {
    TotalTransactions int64   `json:"totalTransactions"`
    SuccessfulTransactions int64 `json:"successfulTransactions"`
    FailedTransactions int64 `json:"failedTransactions"`
    AverageResponseTime string `json:"averageResponseTime"`
    SuccessRate float64 `json:"successRate"`
}

// +kubebuilder:object:root=true
type PaymentProcessorList struct {
    metav1.TypeMeta `json:",inline"`
    metav1.ListMeta `json:"metadata,omitempty"`
    Items           []PaymentProcessor `json:"items"`
}

// Database CRD for advanced database management
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=db
type Database struct {
    metav1.TypeMeta   `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`

    Spec   DatabaseSpec   `json:"spec,omitempty"`
    Status DatabaseStatus `json:"status,omitempty"`
}

type DatabaseSpec struct {
    // Engine defines the database engine type
    // +kubebuilder:validation:Enum=postgresql;mysql;mongodb;redis;cassandra
    Engine string `json:"engine"`

    // Version defines the database version
    Version string `json:"version"`

    // Storage configuration
    Storage StorageConfig `json:"storage"`

    // Backup configuration
    Backup *BackupConfig `json:"backup,omitempty"`

    // High availability configuration
    HighAvailability *HAConfig `json:"highAvailability,omitempty"`

    // Security configuration
    Security *DatabaseSecurityConfig `json:"security,omitempty"`

    // Performance tuning
    Performance *PerformanceConfig `json:"performance,omitempty"`

    // Monitoring configuration
    Monitoring *DatabaseMonitoringConfig `json:"monitoring,omitempty"`
}

type StorageConfig struct {
    // Size defines the storage size
    // +kubebuilder:validation:Pattern=^([+-]?[0-9.]+)([eEinumkKMGTP]*[-+]?[0-9]*)$
    Size string `json:"size"`

    // StorageClass defines the storage class
    StorageClass *string `json:"storageClass,omitempty"`

    // Encryption at rest
    EncryptionAtRest bool `json:"encryptionAtRest,omitempty"`

    // IOPS configuration for high-performance storage
    IOPS *int32 `json:"iops,omitempty"`
}

type HAConfig struct {
    // Replicas defines the number of database replicas
    // +kubebuilder:validation:Minimum=1
    // +kubebuilder:validation:Maximum=10
    Replicas int32 `json:"replicas"`

    // Replication mode
    // +kubebuilder:validation:Enum=sync;async;semi-sync
    ReplicationMode string `json:"replicationMode,omitempty"`

    // Anti-affinity rules
    AntiAffinity *AntiAffinityConfig `json:"antiAffinity,omitempty"`

    // Failover configuration
    Failover *FailoverConfig `json:"failover,omitempty"`
}

// Comprehensive validation webhooks
func (pp *PaymentProcessor) ValidateCreate() error {
    return pp.validatePaymentProcessor()
}

func (pp *PaymentProcessor) ValidateUpdate(old runtime.Object) error {
    oldPP := old.(*PaymentProcessor)
    
    // Prevent downscaling below minimum replicas during high load
    if pp.isHighLoad() && pp.Spec.Replicas < oldPP.Spec.Replicas {
        return fmt.Errorf("cannot downscale during high load period")
    }
    
    return pp.validatePaymentProcessor()
}

func (pp *PaymentProcessor) validatePaymentProcessor() error {
    // Validate banking partners
    if len(pp.Spec.Config.BankingPartners) == 0 {
        return fmt.Errorf("at least one banking partner must be configured")
    }

    // Validate total weight
    totalWeight := int32(0)
    for _, partner := range pp.Spec.Config.BankingPartners {
        totalWeight += partner.Weight
    }
    if totalWeight == 0 {
        return fmt.Errorf("total weight of banking partners must be greater than 0")
    }

    // Validate resource requirements
    if pp.Spec.Resources.Requests.Memory == "" {
        return fmt.Errorf("memory request must be specified")
    }

    return nil
}

// Default values mutation webhook
func (pp *PaymentProcessor) Default() {
    if pp.Spec.Replicas == 0 {
        pp.Spec.Replicas = 3
    }

    if pp.Spec.Config.Timeout == "" {
        pp.Spec.Config.Timeout = "30s"
    }

    // Set default circuit breaker configuration
    if pp.Spec.CircuitBreaker == nil {
        pp.Spec.CircuitBreaker = &CircuitBreakerConfig{
            FailureThreshold: 5,
            SuccessThreshold: 3,
            Timeout:          "60s",
        }
    }

    // Set default monitoring configuration
    if pp.Spec.Monitoring == nil {
        pp.Spec.Monitoring = &MonitoringConfig{
            Enabled: true,
            MetricsPort: 8080,
            HealthCheckPath: "/health",
        }
    }
}

// Custom status management
func (pp *PaymentProcessor) UpdateStatus(ctx context.Context, client client.Client) error {
    // Calculate current metrics
    metrics, err := pp.calculateMetrics(ctx)
    if err != nil {
        return fmt.Errorf("failed to calculate metrics: %w", err)
    }

    pp.Status.Metrics = metrics

    // Update banking partner status
    partnerStatus, err := pp.getBankingPartnerStatus(ctx)
    if err != nil {
        return fmt.Errorf("failed to get banking partner status: %w", err)
    }
    pp.Status.BankingPartnerStatus = partnerStatus

    // Determine phase based on conditions
    pp.Status.Phase = pp.determinePhase()

    // Update last update time
    now := metav1.Now()
    pp.Status.LastUpdateTime = &now

    return client.Status().Update(ctx, pp)
}

func (pp *PaymentProcessor) determinePhase() string {
    if pp.Status.ReadyReplicas == 0 {
        return "Pending"
    }
    
    if pp.Status.ReadyReplicas < pp.Spec.Replicas {
        return "Running" // Partially ready
    }
    
    // Check success rate
    if pp.Status.Metrics != nil && pp.Status.Metrics.SuccessRate < 0.95 {
        return "Failed"
    }
    
    return "Running"
}

// Finalizer management
const PaymentProcessorFinalizer = "paymentprocessor.fintech.io/finalizer"

func (pp *PaymentProcessor) SetupFinalizers() {
    controllerutil.AddFinalizer(pp, PaymentProcessorFinalizer)
}

func (pp *PaymentProcessor) CleanupResources(ctx context.Context) error {
    // Gracefully shutdown payment processing
    if err := pp.gracefulShutdown(ctx); err != nil {
        return fmt.Errorf("failed to gracefully shutdown: %w", err)
    }

    // Clean up external resources
    if err := pp.cleanupExternalResources(ctx); err != nil {
        return fmt.Errorf("failed to cleanup external resources: %w", err)
    }

    return nil
}

func (pp *PaymentProcessor) gracefulShutdown(ctx context.Context) error {
    // Implement graceful shutdown logic
    // - Stop accepting new payments
    // - Wait for in-flight transactions to complete
    // - Close connections to banking partners
    
    shutdownTimeout := 30 * time.Second
    shutdownCtx, cancel := context.WithTimeout(ctx, shutdownTimeout)
    defer cancel()

    // Signal shutdown to all instances
    if err := pp.signalShutdown(shutdownCtx); err != nil {
        return err
    }

    // Wait for graceful shutdown or timeout
    for {
        select {
        case <-shutdownCtx.Done():
            return fmt.Errorf("shutdown timeout exceeded")
        default:
            if pp.allInstancesShutdown() {
                return nil
            }
            time.Sleep(1 * time.Second)
        }
    }
}
```

---

## 🤖 **Kubernetes Operators**

### **Advanced Operator Implementation**

```go
package operator

import (
    "context"
    "fmt"
    "time"

    appsv1 "k8s.io/api/apps/v1"
    corev1 "k8s.io/api/core/v1"
    "k8s.io/apimachinery/pkg/api/errors"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/runtime"
    "k8s.io/apimachinery/pkg/types"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/client"
    "sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
    "sigs.k8s.io/controller-runtime/pkg/log"
    "sigs.k8s.io/controller-runtime/pkg/predicate"
)

// PaymentProcessorReconciler reconciles PaymentProcessor objects
type PaymentProcessorReconciler struct {
    client.Client
    Scheme *runtime.Scheme
    
    // Dependencies
    DeploymentManager *DeploymentManager
    ServiceManager    *ServiceManager
    ConfigMapManager  *ConfigMapManager
    SecretManager     *SecretManager
    HPAManager        *HPAManager
    ServiceMonitor    *ServiceMonitorManager
    
    // Metrics and monitoring
    Metrics *OperatorMetrics
    
    // External integrations
    BankingPartnerValidator *BankingPartnerValidator
    EncryptionService       *EncryptionService
}

// +kubebuilder:rbac:groups=fintech.io,resources=paymentprocessors,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=fintech.io,resources=paymentprocessors/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=fintech.io,resources=paymentprocessors/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=secrets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=autoscaling,resources=horizontalpodautoscalers,verbs=get;list;watch;create;update;patch;delete

func (r *PaymentProcessorReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    log := log.FromContext(ctx)
    
    // Fetch the PaymentProcessor instance
    var paymentProcessor PaymentProcessor
    if err := r.Get(ctx, req.NamespacedName, &paymentProcessor); err != nil {
        if errors.IsNotFound(err) {
            log.Info("PaymentProcessor resource not found. Ignoring since object must be deleted")
            return ctrl.Result{}, nil
        }
        log.Error(err, "Failed to get PaymentProcessor")
        return ctrl.Result{}, err
    }

    // Handle deletion
    if paymentProcessor.GetDeletionTimestamp() != nil {
        return r.handleDeletion(ctx, &paymentProcessor)
    }

    // Add finalizer if not present
    if !controllerutil.ContainsFinalizer(&paymentProcessor, PaymentProcessorFinalizer) {
        controllerutil.AddFinalizer(&paymentProcessor, PaymentProcessorFinalizer)
        if err := r.Update(ctx, &paymentProcessor); err != nil {
            return ctrl.Result{}, err
        }
    }

    // Validate banking partners
    if err := r.validateBankingPartners(ctx, &paymentProcessor); err != nil {
        return r.updateStatusWithError(ctx, &paymentProcessor, err)
    }

    // Reconcile ConfigMap
    if err := r.reconcileConfigMap(ctx, &paymentProcessor); err != nil {
        return r.updateStatusWithError(ctx, &paymentProcessor, err)
    }

    // Reconcile Secrets
    if err := r.reconcileSecrets(ctx, &paymentProcessor); err != nil {
        return r.updateStatusWithError(ctx, &paymentProcessor, err)
    }

    // Reconcile Service
    if err := r.reconcileService(ctx, &paymentProcessor); err != nil {
        return r.updateStatusWithError(ctx, &paymentProcessor, err)
    }

    // Reconcile Deployment
    if err := r.reconcileDeployment(ctx, &paymentProcessor); err != nil {
        return r.updateStatusWithError(ctx, &paymentProcessor, err)
    }

    // Reconcile HPA if autoscaling is enabled
    if paymentProcessor.Spec.Autoscaling != nil {
        if err := r.reconcileHPA(ctx, &paymentProcessor); err != nil {
            return r.updateStatusWithError(ctx, &paymentProcessor, err)
        }
    }

    // Reconcile ServiceMonitor for monitoring
    if paymentProcessor.Spec.Monitoring != nil && paymentProcessor.Spec.Monitoring.Enabled {
        if err := r.reconcileServiceMonitor(ctx, &paymentProcessor); err != nil {
            return r.updateStatusWithError(ctx, &paymentProcessor, err)
        }
    }

    // Update status
    if err := r.updateStatus(ctx, &paymentProcessor); err != nil {
        return ctrl.Result{}, err
    }

    // Record metrics
    r.Metrics.RecordReconciliation(paymentProcessor.Name, paymentProcessor.Namespace, "success")

    return ctrl.Result{RequeueAfter: time.Minute * 5}, nil
}

func (r *PaymentProcessorReconciler) reconcileDeployment(ctx context.Context, pp *PaymentProcessor) error {
    deployment := &appsv1.Deployment{}
    deploymentName := types.NamespacedName{
        Name:      pp.Name,
        Namespace: pp.Namespace,
    }

    err := r.Get(ctx, deploymentName, deployment)
    if err != nil && errors.IsNotFound(err) {
        // Deployment doesn't exist, create it
        deployment = r.DeploymentManager.BuildDeployment(pp)
        if err := controllerutil.SetControllerReference(pp, deployment, r.Scheme); err != nil {
            return fmt.Errorf("failed to set controller reference: %w", err)
        }
        
        if err := r.Create(ctx, deployment); err != nil {
            return fmt.Errorf("failed to create deployment: %w", err)
        }
        
        return nil
    } else if err != nil {
        return fmt.Errorf("failed to get deployment: %w", err)
    }

    // Deployment exists, check if update is needed
    desiredDeployment := r.DeploymentManager.BuildDeployment(pp)
    
    if r.DeploymentManager.NeedsUpdate(deployment, desiredDeployment) {
        // Update deployment
        deployment.Spec = desiredDeployment.Spec
        deployment.ObjectMeta.Labels = desiredDeployment.ObjectMeta.Labels
        deployment.ObjectMeta.Annotations = desiredDeployment.ObjectMeta.Annotations

        if err := r.Update(ctx, deployment); err != nil {
            return fmt.Errorf("failed to update deployment: %w", err)
        }
    }

    return nil
}

func (r *PaymentProcessorReconciler) updateStatus(ctx context.Context, pp *PaymentProcessor) error {
    // Get current deployment status
    deployment := &appsv1.Deployment{}
    deploymentName := types.NamespacedName{
        Name:      pp.Name,
        Namespace: pp.Namespace,
    }

    if err := r.Get(ctx, deploymentName, deployment); err != nil {
        return fmt.Errorf("failed to get deployment for status update: %w", err)
    }

    // Update replicas status
    pp.Status.Replicas = deployment.Status.Replicas
    pp.Status.ReadyReplicas = deployment.Status.ReadyReplicas

    // Update conditions
    conditions := []metav1.Condition{}
    
    // Deployment Available condition
    if deployment.Status.Conditions != nil {
        for _, deploymentCondition := range deployment.Status.Conditions {
            if deploymentCondition.Type == appsv1.DeploymentAvailable {
                condition := metav1.Condition{
                    Type:    "Available",
                    Status:  metav1.ConditionStatus(deploymentCondition.Status),
                    Reason:  deploymentCondition.Reason,
                    Message: deploymentCondition.Message,
                    LastTransitionTime: deploymentCondition.LastTransitionTime,
                }
                conditions = append(conditions, condition)
                break
            }
        }
    }

    // Banking partners health condition
    bankingHealthy, bankingMessage := r.checkBankingPartnersHealth(ctx, pp)
    bankingCondition := metav1.Condition{
        Type:   "BankingPartnersHealthy",
        Status: metav1.ConditionTrue,
        Reason: "BankingPartnersHealthy",
        Message: "All banking partners are healthy",
        LastTransitionTime: metav1.Now(),
    }
    if !bankingHealthy {
        bankingCondition.Status = metav1.ConditionFalse
        bankingCondition.Reason = "BankingPartnersUnhealthy"
        bankingCondition.Message = bankingMessage
    }
    conditions = append(conditions, bankingCondition)

    pp.Status.Conditions = conditions

    // Calculate and update metrics
    metrics, err := r.calculateProcessorMetrics(ctx, pp)
    if err == nil {
        pp.Status.Metrics = metrics
    }

    // Update phase
    pp.Status.Phase = r.determinePhase(pp)

    // Update timestamp
    now := metav1.Now()
    pp.Status.LastUpdateTime = &now

    return r.Status().Update(ctx, pp)
}

// DeploymentManager handles deployment creation and updates
type DeploymentManager struct {
    ImageRegistry string
    ImageTag      string
}

func (dm *DeploymentManager) BuildDeployment(pp *PaymentProcessor) *appsv1.Deployment {
    labels := map[string]string{
        "app":                          pp.Name,
        "app.kubernetes.io/name":       "payment-processor",
        "app.kubernetes.io/instance":   pp.Name,
        "app.kubernetes.io/component":  pp.Spec.ProcessorType,
        "app.kubernetes.io/part-of":    "payment-system",
        "app.kubernetes.io/managed-by": "payment-processor-operator",
    }

    deployment := &appsv1.Deployment{
        ObjectMeta: metav1.ObjectMeta{
            Name:      pp.Name,
            Namespace: pp.Namespace,
            Labels:    labels,
            Annotations: map[string]string{
                "deployment.kubernetes.io/revision": "1",
                "fintech.io/processor-type":         pp.Spec.ProcessorType,
            },
        },
        Spec: appsv1.DeploymentSpec{
            Replicas: &pp.Spec.Replicas,
            Selector: &metav1.LabelSelector{
                MatchLabels: labels,
            },
            Strategy: appsv1.DeploymentStrategy{
                Type: appsv1.RollingUpdateDeploymentStrategyType,
                RollingUpdate: &appsv1.RollingUpdateDeployment{
                    MaxUnavailable: &intstr.FromString("25%"),
                    MaxSurge:       &intstr.FromString("25%"),
                },
            },
            Template: corev1.PodTemplateSpec{
                ObjectMeta: metav1.ObjectMeta{
                    Labels: labels,
                    Annotations: map[string]string{
                        "prometheus.io/scrape": "true",
                        "prometheus.io/port":   "8080",
                        "prometheus.io/path":   "/metrics",
                    },
                },
                Spec: dm.buildPodSpec(pp),
            },
        },
    }

    return deployment
}

func (dm *DeploymentManager) buildPodSpec(pp *PaymentProcessor) corev1.PodSpec {
    podSpec := corev1.PodSpec{
        Containers: []corev1.Container{
            {
                Name:  "payment-processor",
                Image: fmt.Sprintf("%s/payment-processor:%s", dm.ImageRegistry, dm.ImageTag),
                Ports: []corev1.ContainerPort{
                    {
                        Name:          "http",
                        ContainerPort: 8080,
                        Protocol:      corev1.ProtocolTCP,
                    },
                    {
                        Name:          "metrics",
                        ContainerPort: 8081,
                        Protocol:      corev1.ProtocolTCP,
                    },
                },
                Env: dm.buildEnvironmentVariables(pp),
                Resources: corev1.ResourceRequirements{
                    Requests: pp.Spec.Resources.Requests,
                    Limits:   pp.Spec.Resources.Limits,
                },
                LivenessProbe: &corev1.Probe{
                    ProbeHandler: corev1.ProbeHandler{
                        HTTPGet: &corev1.HTTPGetAction{
                            Path: "/health/live",
                            Port: intstr.FromInt(8080),
                        },
                    },
                    InitialDelaySeconds: 30,
                    PeriodSeconds:       10,
                    TimeoutSeconds:      5,
                    FailureThreshold:    3,
                },
                ReadinessProbe: &corev1.Probe{
                    ProbeHandler: corev1.ProbeHandler{
                        HTTPGet: &corev1.HTTPGetAction{
                            Path: "/health/ready",
                            Port: intstr.FromInt(8080),
                        },
                    },
                    InitialDelaySeconds: 5,
                    PeriodSeconds:       5,
                    TimeoutSeconds:      3,
                    FailureThreshold:    3,
                },
                VolumeMounts: []corev1.VolumeMount{
                    {
                        Name:      "config",
                        MountPath: "/etc/payment-processor",
                        ReadOnly:  true,
                    },
                    {
                        Name:      "secrets",
                        MountPath: "/etc/secrets",
                        ReadOnly:  true,
                    },
                },
            },
        },
        Volumes: []corev1.Volume{
            {
                Name: "config",
                VolumeSource: corev1.VolumeSource{
                    ConfigMap: &corev1.ConfigMapVolumeSource{
                        LocalObjectReference: corev1.LocalObjectReference{
                            Name: pp.Name + "-config",
                        },
                    },
                },
            },
            {
                Name: "secrets",
                VolumeSource: corev1.VolumeSource{
                    Secret: &corev1.SecretVolumeSource{
                        SecretName: pp.Name + "-secrets",
                    },
                },
            },
        },
        ServiceAccountName: pp.Name,
        SecurityContext:    pp.Spec.SecurityContext,
    }

    // Add anti-affinity rules for high availability
    if pp.Spec.Replicas > 1 {
        podSpec.Affinity = &corev1.Affinity{
            PodAntiAffinity: &corev1.PodAntiAffinity{
                PreferredDuringSchedulingIgnoredDuringExecution: []corev1.WeightedPodAffinityTerm{
                    {
                        Weight: 100,
                        PodAffinityTerm: corev1.PodAffinityTerm{
                            LabelSelector: &metav1.LabelSelector{
                                MatchLabels: map[string]string{
                                    "app": pp.Name,
                                },
                            },
                            TopologyKey: "kubernetes.io/hostname",
                        },
                    },
                },
            },
        }
    }

    return podSpec
}

func (dm *DeploymentManager) buildEnvironmentVariables(pp *PaymentProcessor) []corev1.EnvVar {
    env := []corev1.EnvVar{
        {
            Name:  "PROCESSOR_TYPE",
            Value: pp.Spec.ProcessorType,
        },
        {
            Name:  "TIMEOUT",
            Value: pp.Spec.Config.Timeout,
        },
        {
            Name: "POD_NAME",
            ValueFrom: &corev1.EnvVarSource{
                FieldRef: &corev1.ObjectFieldSelector{
                    FieldPath: "metadata.name",
                },
            },
        },
        {
            Name: "POD_NAMESPACE",
            ValueFrom: &corev1.EnvVarSource{
                FieldRef: &corev1.ObjectFieldSelector{
                    FieldPath: "metadata.namespace",
                },
            },
        },
        {
            Name: "POD_IP",
            ValueFrom: &corev1.EnvVarSource{
                FieldRef: &corev1.ObjectFieldSelector{
                    FieldPath: "status.podIP",
                },
            },
        },
    }

    // Add circuit breaker configuration
    if pp.Spec.CircuitBreaker != nil {
        env = append(env, []corev1.EnvVar{
            {
                Name:  "CIRCUIT_BREAKER_FAILURE_THRESHOLD",
                Value: fmt.Sprintf("%d", pp.Spec.CircuitBreaker.FailureThreshold),
            },
            {
                Name:  "CIRCUIT_BREAKER_SUCCESS_THRESHOLD",
                Value: fmt.Sprintf("%d", pp.Spec.CircuitBreaker.SuccessThreshold),
            },
            {
                Name:  "CIRCUIT_BREAKER_TIMEOUT",
                Value: pp.Spec.CircuitBreaker.Timeout,
            },
        }...)
    }

    // Add rate limiting configuration
    if pp.Spec.Config.RateLimit != nil {
        env = append(env, []corev1.EnvVar{
            {
                Name:  "RATE_LIMIT_RPS",
                Value: fmt.Sprintf("%d", pp.Spec.Config.RateLimit.RequestsPerSecond),
            },
            {
                Name:  "RATE_LIMIT_BURST",
                Value: fmt.Sprintf("%d", pp.Spec.Config.RateLimit.BurstSize),
            },
        }...)
    }

    return env
}

// SetupWithManager sets up the controller with the Manager
func (r *PaymentProcessorReconciler) SetupWithManager(mgr ctrl.Manager) error {
    return ctrl.NewControllerManagedBy(mgr).
        For(&PaymentProcessor{}).
        Owns(&appsv1.Deployment{}).
        Owns(&corev1.Service{}).
        Owns(&corev1.ConfigMap{}).
        Owns(&corev1.Secret{}).
        WithOptions(controller.Options{
            MaxConcurrentReconciles: 5,
        }).
        WithEventFilter(predicate.GenerationChangedPredicate{}).
        Complete(r)
}

// Advanced operator features
type OperatorFeatures struct {
    // Health checking
    HealthChecker *HealthChecker
    
    // Auto-scaling based on business metrics
    BusinessMetricsScaler *BusinessMetricsScaler
    
    // Chaos engineering integration
    ChaosEngineer *ChaosEngineer
    
    // Cost optimization
    CostOptimizer *CostOptimizer
    
    // Performance tuning
    PerformanceTuner *PerformanceTuner
}

// BusinessMetricsScaler scales based on payment processing metrics
type BusinessMetricsScaler struct {
    MetricsClient    MetricsClient
    ScalingPolicies  []ScalingPolicy
    CooldownPeriod   time.Duration
    LastScaleTime    time.Time
}

type ScalingPolicy struct {
    MetricName    string    `json:"metricName"`
    Threshold     float64   `json:"threshold"`
    ScaleAction   string    `json:"scaleAction"`  // "scale_up" or "scale_down"
    ScaleFactor   float64   `json:"scaleFactor"`
    MinReplicas   int32     `json:"minReplicas"`
    MaxReplicas   int32     `json:"maxReplicas"`
}

func (bms *BusinessMetricsScaler) EvaluateScaling(ctx context.Context, pp *PaymentProcessor) (int32, error) {
    // Check cooldown period
    if time.Since(bms.LastScaleTime) < bms.CooldownPeriod {
        return pp.Status.Replicas, nil
    }

    currentReplicas := pp.Status.Replicas
    targetReplicas := currentReplicas

    for _, policy := range bms.ScalingPolicies {
        metricValue, err := bms.MetricsClient.GetMetric(ctx, policy.MetricName, pp.Name, pp.Namespace)
        if err != nil {
            continue
        }

        if policy.ScaleAction == "scale_up" && metricValue > policy.Threshold {
            newReplicas := int32(float64(currentReplicas) * policy.ScaleFactor)
            if newReplicas > policy.MaxReplicas {
                newReplicas = policy.MaxReplicas
            }
            if newReplicas > targetReplicas {
                targetReplicas = newReplicas
            }
        } else if policy.ScaleAction == "scale_down" && metricValue < policy.Threshold {
            newReplicas := int32(float64(currentReplicas) / policy.ScaleFactor)
            if newReplicas < policy.MinReplicas {
                newReplicas = policy.MinReplicas
            }
            if newReplicas < targetReplicas {
                targetReplicas = newReplicas
            }
        }
    }

    if targetReplicas != currentReplicas {
        bms.LastScaleTime = time.Now()
    }

    return targetReplicas, nil
}
```

This comprehensive Kubernetes guide demonstrates advanced CRD and operator patterns essential for managing complex distributed systems at scale. The implementations show production-ready patterns for payment processing systems with proper validation, status management, and advanced operator features.

---

## 🌐 **Advanced Networking & Service Mesh**

### **Istio Service Mesh Implementation**

```go
package servicemesh

import (
    "context"
    "fmt"
    "time"

    istionetworking "istio.io/api/networking/v1beta1"
    istiosecurity "istio.io/api/security/v1beta1"
    istioclient "istio.io/client-go/pkg/clientset/versioned"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "sigs.k8s.io/controller-runtime/pkg/client"
)

// ServiceMeshManager manages Istio service mesh configurations
type ServiceMeshManager struct {
    IstioClient    istioclient.Interface
    K8sClient      client.Client
    Namespace      string
    
    // Traffic management
    TrafficManager *TrafficManager
    
    // Security policies
    SecurityManager *SecurityPolicyManager
    
    // Observability
    ObservabilityManager *ObservabilityManager
    
    // Fault injection
    ChaosManager *ChaosManager
}

// Traffic management for payment processing
type TrafficManager struct {
    client         istioclient.Interface
    routingRules   []RoutingRule
    loadBalancers  []LoadBalancingRule
    circuitBreakers []CircuitBreakerRule
}

type RoutingRule struct {
    ServiceName    string            `json:"serviceName"`
    Routes         []RouteRule       `json:"routes"`
    FaultInjection *FaultInjection   `json:"faultInjection,omitempty"`
    Timeout        *time.Duration    `json:"timeout,omitempty"`
    Retries        *RetryPolicy      `json:"retries,omitempty"`
}

type RouteRule struct {
    Match       []HTTPMatchRequest `json:"match"`
    Route       []HTTPRouteDestination `json:"route"`
    Headers     *Headers           `json:"headers,omitempty"`
    Redirect    *HTTPRedirect      `json:"redirect,omitempty"`
}

func (tm *TrafficManager) CreateVirtualService(ctx context.Context, rule RoutingRule) error {
    // Create Istio VirtualService for advanced traffic routing
    vs := &istionetworking.VirtualService{
        ObjectMeta: metav1.ObjectMeta{
            Name:      fmt.Sprintf("%s-vs", rule.ServiceName),
            Namespace: "default",
            Labels: map[string]string{
                "app":     rule.ServiceName,
                "version": "v1",
            },
        },
        Spec: istionetworking.VirtualService{
            Hosts: []string{rule.ServiceName},
            Http:  tm.buildHTTPRoutes(rule),
        },
    }

    // Add fault injection if specified
    if rule.FaultInjection != nil {
        vs.Spec.Http[0].Fault = &istionetworking.HTTPFaultInjection{
            Delay: &istionetworking.HTTPFaultInjection_Delay{
                HttpDelayType: &istionetworking.HTTPFaultInjection_Delay_FixedDelay{
                    FixedDelay: &rule.FaultInjection.Delay,
                },
                Percentage: &istionetworking.Percent{
                    Value: rule.FaultInjection.Percentage,
                },
            },
        }
    }

    _, err := tm.client.NetworkingV1beta1().VirtualServices("default").Create(ctx, vs, metav1.CreateOptions{})
    return err
}

// Payment processor destination rule with circuit breaker
func (tm *TrafficManager) CreateDestinationRule(ctx context.Context, serviceName string, cbRule CircuitBreakerRule) error {
    dr := &istionetworking.DestinationRule{
        ObjectMeta: metav1.ObjectMeta{
            Name:      fmt.Sprintf("%s-dr", serviceName),
            Namespace: "default",
        },
        Spec: istionetworking.DestinationRule{
            Host: serviceName,
            TrafficPolicy: &istionetworking.TrafficPolicy{
                LoadBalancer: &istionetworking.LoadBalancerSettings{
                    LbPolicy: &istionetworking.LoadBalancerSettings_Simple{
                        Simple: istionetworking.LoadBalancerSettings_ROUND_ROBIN,
                    },
                },
                ConnectionPool: &istionetworking.ConnectionPoolSettings{
                    Tcp: &istionetworking.ConnectionPoolSettings_TCPSettings{
                        MaxConnections: int32(cbRule.MaxConnections),
                        ConnectTimeout: &cbRule.ConnectTimeout,
                    },
                    Http: &istionetworking.ConnectionPoolSettings_HTTPSettings{
                        Http1MaxPendingRequests:  int32(cbRule.MaxPendingRequests),
                        Http2MaxRequests:         int32(cbRule.MaxRequests),
                        MaxRequestsPerConnection: int32(cbRule.MaxRequestsPerConnection),
                        MaxRetries:              int32(cbRule.MaxRetries),
                        IdleTimeout:             &cbRule.IdleTimeout,
                    },
                },
                OutlierDetection: &istionetworking.OutlierDetection{
                    ConsecutiveGatewayErrors: &cbRule.ConsecutiveErrors,
                    Interval:                &cbRule.Interval,
                    BaseEjectionTime:        &cbRule.BaseEjectionTime,
                    MaxEjectionPercent:      int32(cbRule.MaxEjectionPercent),
                    MinHealthPercent:        int32(cbRule.MinHealthPercent),
                },
            },
            Subsets: []istionetworking.Subset{
                {
                    Name: "v1",
                    Labels: map[string]string{
                        "version": "v1",
                    },
                },
                {
                    Name: "v2",
                    Labels: map[string]string{
                        "version": "v2",
                    },
                },
            },
        },
    }

    _, err := tm.client.NetworkingV1beta1().DestinationRules("default").Create(ctx, dr, metav1.CreateOptions{})
    return err
}

// Canary deployment with traffic splitting
func (tm *TrafficManager) SetupCanaryDeployment(ctx context.Context, service string, canaryWeight int32) error {
    vs := &istionetworking.VirtualService{
        ObjectMeta: metav1.ObjectMeta{
            Name:      fmt.Sprintf("%s-canary", service),
            Namespace: "default",
        },
        Spec: istionetworking.VirtualService{
            Hosts: []string{service},
            Http: []*istionetworking.HTTPRoute{
                {
                    Match: []*istionetworking.HTTPMatchRequest{
                        {
                            Headers: map[string]*istionetworking.StringMatch{
                                "canary": {
                                    MatchType: &istionetworking.StringMatch_Exact{
                                        Exact: "true",
                                    },
                                },
                            },
                        },
                    },
                    Route: []*istionetworking.HTTPRouteDestination{
                        {
                            Destination: &istionetworking.Destination{
                                Host:   service,
                                Subset: "v2",
                            },
                            Weight: 100,
                        },
                    },
                },
                {
                    Route: []*istionetworking.HTTPRouteDestination{
                        {
                            Destination: &istionetworking.Destination{
                                Host:   service,
                                Subset: "v1",
                            },
                            Weight: 100 - canaryWeight,
                        },
                        {
                            Destination: &istionetworking.Destination{
                                Host:   service,
                                Subset: "v2",
                            },
                            Weight: canaryWeight,
                        },
                    },
                },
            },
        },
    }

    _, err := tm.client.NetworkingV1beta1().VirtualServices("default").Create(ctx, vs, metav1.CreateOptions{})
    return err
}

// Security policies for service mesh
type SecurityPolicyManager struct {
    client istioclient.Interface
}

// Mutual TLS policy for payment services
func (spm *SecurityPolicyManager) CreateMTLSPolicy(ctx context.Context, namespace string) error {
    policy := &istiosecurity.PeerAuthentication{
        ObjectMeta: metav1.ObjectMeta{
            Name:      "default",
            Namespace: namespace,
        },
        Spec: istiosecurity.PeerAuthentication{
            Mtls: &istiosecurity.PeerAuthentication_MutualTLS{
                Mode: istiosecurity.PeerAuthentication_MutualTLS_STRICT,
            },
        },
    }

    _, err := spm.client.SecurityV1beta1().PeerAuthentications(namespace).Create(ctx, policy, metav1.CreateOptions{})
    return err
}

// Authorization policy for payment endpoints
func (spm *SecurityPolicyManager) CreateAuthorizationPolicy(ctx context.Context, serviceName, namespace string) error {
    policy := &istiosecurity.AuthorizationPolicy{
        ObjectMeta: metav1.ObjectMeta{
            Name:      fmt.Sprintf("%s-authz", serviceName),
            Namespace: namespace,
        },
        Spec: istiosecurity.AuthorizationPolicy{
            Selector: &istiosecurity.WorkloadSelector{
                MatchLabels: map[string]string{
                    "app": serviceName,
                },
            },
            Rules: []*istiosecurity.Rule{
                {
                    From: []*istiosecurity.Rule_From{
                        {
                            Source: &istiosecurity.Source{
                                Principals: []string{
                                    "cluster.local/ns/payment-system/sa/payment-gateway",
                                },
                            },
                        },
                    },
                    To: []*istiosecurity.Rule_To{
                        {
                            Operation: &istiosecurity.Operation{
                                Methods: []string{"POST", "GET"},
                                Paths:   []string{"/api/v1/payments/*"},
                            },
                        },
                    },
                    When: []*istiosecurity.Condition{
                        {
                            Key:    "request.headers[authorization]",
                            Values: []string{"Bearer *"},
                        },
                    },
                },
            },
        },
    }

    _, err := spm.client.SecurityV1beta1().AuthorizationPolicies(namespace).Create(ctx, policy, metav1.CreateOptions{})
    return err
}

// Gateway configuration for external traffic
func (sm *ServiceMeshManager) CreatePaymentGateway(ctx context.Context) error {
    gateway := &istionetworking.Gateway{
        ObjectMeta: metav1.ObjectMeta{
            Name:      "payment-gateway",
            Namespace: "istio-system",
        },
        Spec: istionetworking.Gateway{
            Selector: map[string]string{
                "istio": "ingressgateway",
            },
            Servers: []*istionetworking.Server{
                {
                    Port: &istionetworking.Port{
                        Number:   443,
                        Name:     "https",
                        Protocol: "HTTPS",
                    },
                    Hosts: []string{"payments.razorpay.com"},
                    Tls: &istionetworking.ServerTLSSettings{
                        Mode:           istionetworking.ServerTLSSettings_SIMPLE,
                        CredentialName: "payment-tls-secret",
                    },
                },
                {
                    Port: &istionetworking.Port{
                        Number:   80,
                        Name:     "http",
                        Protocol: "HTTP",
                    },
                    Hosts: []string{"payments.razorpay.com"},
                    Tls: &istionetworking.ServerTLSSettings{
                        HttpsRedirect: true,
                    },
                },
            },
        },
    }

    _, err := sm.IstioClient.NetworkingV1beta1().Gateways("istio-system").Create(ctx, gateway, metav1.CreateOptions{})
    return err
}
```

---

## 🔐 **Security Policies & RBAC**

### **Advanced Security Implementation**

```go
package security

import (
    "context"
    "fmt"

    corev1 "k8s.io/api/core/v1"
    rbacv1 "k8s.io/api/rbac/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/client-go/kubernetes"
    policyv1beta1 "k8s.io/api/policy/v1beta1"
    netv1 "k8s.io/api/networking/v1"
)

// SecurityManager handles comprehensive security policies
type SecurityManager struct {
    clientset     kubernetes.Interface
    namespace     string
    
    // Policy managers
    RBACManager         *RBACManager
    NetworkPolicyManager *NetworkPolicyManager
    PodSecurityManager  *PodSecurityManager
    ServiceAccountManager *ServiceAccountManager
    SecretsManager      *SecretsManager
}

// RBAC Manager for fine-grained access control
type RBACManager struct {
    clientset kubernetes.Interface
}

// Create comprehensive RBAC for payment processing system
func (rbac *RBACManager) SetupPaymentSystemRBAC(ctx context.Context, namespace string) error {
    // Service Account for payment processor
    sa := &corev1.ServiceAccount{
        ObjectMeta: metav1.ObjectMeta{
            Name:      "payment-processor",
            Namespace: namespace,
            Annotations: map[string]string{
                "eks.amazonaws.com/role-arn": "arn:aws:iam::123456789012:role/PaymentProcessorRole",
            },
        },
        AutomountServiceAccountToken: &[]bool{true}[0],
    }

    if _, err := rbac.clientset.CoreV1().ServiceAccounts(namespace).Create(ctx, sa, metav1.CreateOptions{}); err != nil {
        return fmt.Errorf("failed to create service account: %w", err)
    }

    // Role for payment processor operations
    role := &rbacv1.Role{
        ObjectMeta: metav1.ObjectMeta{
            Name:      "payment-processor-role",
            Namespace: namespace,
        },
        Rules: []rbacv1.PolicyRule{
            {
                APIGroups: [""],
                Resources: ["secrets"],
                ResourceNames: []string{
                    "payment-processor-secrets",
                    "banking-partner-credentials",
                    "encryption-keys",
                },
                Verbs: ["get", "list"],
            },
            {
                APIGroups: [""],
                Resources: ["configmaps"],
                ResourceNames: []string{
                    "payment-processor-config",
                    "routing-rules",
                },
                Verbs: ["get", "list", "watch"],
            },
            {
                APIGroups: [""],
                Resources: ["services", "endpoints"],
                Verbs: ["get", "list", "watch"],
            },
            {
                APIGroups: ["fintech.io"],
                Resources: ["paymentprocessors"],
                Verbs: ["get", "list", "watch", "update", "patch"],
            },
            {
                APIGroups: ["fintech.io"],
                Resources: ["paymentprocessors/status"],
                Verbs: ["get", "update", "patch"],
            },
        },
    }

    if _, err := rbac.clientset.RbacV1().Roles(namespace).Create(ctx, role, metav1.CreateOptions{}); err != nil {
        return fmt.Errorf("failed to create role: %w", err)
    }

    // RoleBinding
    roleBinding := &rbacv1.RoleBinding{
        ObjectMeta: metav1.ObjectMeta{
            Name:      "payment-processor-binding",
            Namespace: namespace,
        },
        Subjects: []rbacv1.Subject{
            {
                Kind:      "ServiceAccount",
                Name:      "payment-processor",
                Namespace: namespace,
            },
        },
        RoleRef: rbacv1.RoleRef{
            APIGroup: "rbac.authorization.k8s.io",
            Kind:     "Role",
            Name:     "payment-processor-role",
        },
    }

    if _, err := rbac.clientset.RbacV1().RoleBindings(namespace).Create(ctx, roleBinding, metav1.CreateOptions{}); err != nil {
        return fmt.Errorf("failed to create role binding: %w", err)
    }

    return nil
}

// Cluster-level RBAC for operators
func (rbac *RBACManager) SetupOperatorRBAC(ctx context.Context) error {
    // ClusterRole for payment processor operator
    clusterRole := &rbacv1.ClusterRole{
        ObjectMeta: metav1.ObjectMeta{
            Name: "payment-processor-operator",
        },
        Rules: []rbacv1.PolicyRule{
            {
                APIGroups: ["fintech.io"],
                Resources: ["paymentprocessors", "databases"],
                Verbs: ["*"],
            },
            {
                APIGroups: ["apps"],
                Resources: ["deployments", "replicasets"],
                Verbs: ["*"],
            },
            {
                APIGroups: [""],
                Resources: ["services", "configmaps", "secrets", "serviceaccounts"],
                Verbs: ["*"],
            },
            {
                APIGroups: ["autoscaling"],
                Resources: ["horizontalpodautoscalers"],
                Verbs: ["*"],
            },
            {
                APIGroups: ["monitoring.coreos.com"],
                Resources: ["servicemonitors"],
                Verbs: ["*"],
            },
            {
                APIGroups: ["networking.k8s.io"],
                Resources: ["networkpolicies"],
                Verbs: ["*"],
            },
            {
                APIGroups: ["policy"],
                Resources: ["poddisruptionbudgets"],
                Verbs: ["*"],
            },
        },
    }

    if _, err := rbac.clientset.RbacV1().ClusterRoles().Create(ctx, clusterRole, metav1.CreateOptions{}); err != nil {
        return fmt.Errorf("failed to create cluster role: %w", err)
    }

    return nil
}

// Network Policy Manager for traffic isolation
type NetworkPolicyManager struct {
    clientset kubernetes.Interface
}

// Create network policies for payment system isolation
func (npm *NetworkPolicyManager) CreatePaymentNetworkPolicies(ctx context.Context, namespace string) error {
    // Default deny all ingress policy
    denyAllPolicy := &netv1.NetworkPolicy{
        ObjectMeta: metav1.ObjectMeta{
            Name:      "deny-all-ingress",
            Namespace: namespace,
        },
        Spec: netv1.NetworkPolicySpec{
            PodSelector: metav1.LabelSelector{},
            PolicyTypes: []netv1.PolicyType{netv1.PolicyTypeIngress},
        },
    }

    if _, err := npm.clientset.NetworkingV1().NetworkPolicies(namespace).Create(ctx, denyAllPolicy, metav1.CreateOptions{}); err != nil {
        return fmt.Errorf("failed to create deny-all policy: %w", err)
    }

    // Allow payment processor to banking partners
    allowBankingPolicy := &netv1.NetworkPolicy{
        ObjectMeta: metav1.ObjectMeta{
            Name:      "allow-payment-to-banking",
            Namespace: namespace,
        },
        Spec: netv1.NetworkPolicySpec{
            PodSelector: metav1.LabelSelector{
                MatchLabels: map[string]string{
                    "app": "payment-processor",
                },
            },
            PolicyTypes: []netv1.PolicyType{netv1.PolicyTypeEgress},
            Egress: []netv1.NetworkPolicyEgressRule{
                {
                    To: []netv1.NetworkPolicyPeer{
                        {
                            PodSelector: &metav1.LabelSelector{
                                MatchLabels: map[string]string{
                                    "component": "banking-gateway",
                                },
                            },
                        },
                    },
                    Ports: []netv1.NetworkPolicyPort{
                        {
                            Protocol: &[]corev1.Protocol{corev1.ProtocolTCP}[0],
                            Port:     &intstr.FromInt(8080),
                        },
                        {
                            Protocol: &[]corev1.Protocol{corev1.ProtocolTCP}[0],
                            Port:     &intstr.FromInt(443),
                        },
                    },
                },
                // Allow DNS resolution
                {
                    To: []netv1.NetworkPolicyPeer{
                        {
                            NamespaceSelector: &metav1.LabelSelector{
                                MatchLabels: map[string]string{
                                    "name": "kube-system",
                                },
                            },
                        },
                    },
                    Ports: []netv1.NetworkPolicyPort{
                        {
                            Protocol: &[]corev1.Protocol{corev1.ProtocolUDP}[0],
                            Port:     &intstr.FromInt(53),
                        },
                    },
                },
            },
        },
    }

    if _, err := npm.clientset.NetworkingV1().NetworkPolicies(namespace).Create(ctx, allowBankingPolicy, metav1.CreateOptions{}); err != nil {
        return fmt.Errorf("failed to create banking policy: %w", err)
    }

    // Allow ingress from API gateway
    allowIngressPolicy := &netv1.NetworkPolicy{
        ObjectMeta: metav1.ObjectMeta{
            Name:      "allow-api-gateway-ingress",
            Namespace: namespace,
        },
        Spec: netv1.NetworkPolicySpec{
            PodSelector: metav1.LabelSelector{
                MatchLabels: map[string]string{
                    "app": "payment-processor",
                },
            },
            PolicyTypes: []netv1.PolicyType{netv1.PolicyTypeIngress},
            Ingress: []netv1.NetworkPolicyIngressRule{
                {
                    From: []netv1.NetworkPolicyPeer{
                        {
                            PodSelector: &metav1.LabelSelector{
                                MatchLabels: map[string]string{
                                    "app": "api-gateway",
                                },
                            },
                        },
                        {
                            NamespaceSelector: &metav1.LabelSelector{
                                MatchLabels: map[string]string{
                                    "name": "istio-system",
                                },
                            },
                            PodSelector: &metav1.LabelSelector{
                                MatchLabels: map[string]string{
                                    "app": "istio-proxy",
                                },
                            },
                        },
                    },
                    Ports: []netv1.NetworkPolicyPort{
                        {
                            Protocol: &[]corev1.Protocol{corev1.ProtocolTCP}[0],
                            Port:     &intstr.FromInt(8080),
                        },
                    },
                },
            },
        },
    }

    if _, err := npm.clientset.NetworkingV1().NetworkPolicies(namespace).Create(ctx, allowIngressPolicy, metav1.CreateOptions{}); err != nil {
        return fmt.Errorf("failed to create ingress policy: %w", err)
    }

    return nil
}

// Pod Security Manager for security contexts and policies
type PodSecurityManager struct {
    clientset kubernetes.Interface
}

// Create Pod Security Policy for payment processors
func (psm *PodSecurityManager) CreatePodSecurityPolicy(ctx context.Context) error {
    psp := &policyv1beta1.PodSecurityPolicy{
        ObjectMeta: metav1.ObjectMeta{
            Name: "payment-processor-psp",
            Annotations: map[string]string{
                "seccomp.security.alpha.kubernetes.io/allowedProfileNames": "runtime/default",
                "apparmor.security.beta.kubernetes.io/allowedProfileNames": "runtime/default",
                "seccomp.security.alpha.kubernetes.io/defaultProfileName":  "runtime/default",
                "apparmor.security.beta.kubernetes.io/defaultProfileName":  "runtime/default",
            },
        },
        Spec: policyv1beta1.PodSecurityPolicySpec{
            Privileged:               false,
            AllowPrivilegeEscalation: &[]bool{false}[0],
            RequiredDropCapabilities: []corev1.Capability{
                "ALL",
            },
            AllowedCapabilities: []corev1.Capability{},
            Volumes: []policyv1beta1.FSType{
                policyv1beta1.ConfigMap,
                policyv1beta1.EmptyDir,
                policyv1beta1.Projected,
                policyv1beta1.Secret,
                policyv1beta1.DownwardAPI,
                policyv1beta1.PersistentVolumeClaim,
            },
            HostNetwork: false,
            HostIPC:     false,
            HostPID:     false,
            RunAsUser: policyv1beta1.RunAsUserStrategyOptions{
                Rule: policyv1beta1.RunAsUserStrategyMustRunAsNonRoot,
            },
            SELinux: policyv1beta1.SELinuxStrategyOptions{
                Rule: policyv1beta1.SELinuxStrategyRunAsAny,
            },
            FSGroup: policyv1beta1.FSGroupStrategyOptions{
                Rule: policyv1beta1.FSGroupStrategyRunAsAny,
            },
            ReadOnlyRootFilesystem: true,
        },
    }

    if _, err := psm.clientset.PolicyV1beta1().PodSecurityPolicies().Create(ctx, psp, metav1.CreateOptions{}); err != nil {
        return fmt.Errorf("failed to create pod security policy: %w", err)
    }

    return nil
}

// Secrets Manager for secure credential management
type SecretsManager struct {
    clientset kubernetes.Interface
}

// Create encrypted secrets for banking partners
func (sm *SecretsManager) CreateBankingPartnerSecrets(ctx context.Context, namespace string, partners []BankingPartnerCredentials) error {
    for _, partner := range partners {
        secret := &corev1.Secret{
            ObjectMeta: metav1.ObjectMeta{
                Name:      fmt.Sprintf("banking-%s-credentials", partner.Name),
                Namespace: namespace,
                Annotations: map[string]string{
                    "encryption-key-id": partner.EncryptionKeyID,
                    "rotation-policy":   "90d",
                },
                Labels: map[string]string{
                    "type":    "banking-credentials",
                    "partner": partner.Name,
                },
            },
            Type: corev1.SecretTypeOpaque,
            Data: map[string][]byte{
                "api-key":        []byte(partner.APIKey),
                "api-secret":     []byte(partner.APISecret),
                "client-cert":    partner.ClientCert,
                "client-key":     partner.ClientKey,
                "ca-cert":        partner.CACert,
                "webhook-secret": []byte(partner.WebhookSecret),
            },
        }

        if _, err := sm.clientset.CoreV1().Secrets(namespace).Create(ctx, secret, metav1.CreateOptions{}); err != nil {
            return fmt.Errorf("failed to create secret for %s: %w", partner.Name, err)
        }
    }

    return nil
}

// Create TLS secrets for service mesh
func (sm *SecretsManager) CreateTLSSecrets(ctx context.Context, namespace string, tlsConfig TLSConfiguration) error {
    secret := &corev1.Secret{
        ObjectMeta: metav1.ObjectMeta{
            Name:      "payment-tls-secret",
            Namespace: namespace,
            Annotations: map[string]string{
                "cert-manager.io/issuer": "letsencrypt-prod",
            },
        },
        Type: corev1.SecretTypeTLS,
        Data: map[string][]byte{
            "tls.crt": tlsConfig.Certificate,
            "tls.key": tlsConfig.PrivateKey,
            "ca.crt":  tlsConfig.CACertificate,
        },
    }

    if _, err := sm.clientset.CoreV1().Secrets(namespace).Create(ctx, secret, metav1.CreateOptions{}); err != nil {
        return fmt.Errorf("failed to create TLS secret: %w", err)
    }

    return nil
}

// Security context factory
func CreateSecurityContext(readOnlyRoot bool, runAsNonRoot bool, userID int64) *corev1.SecurityContext {
    return &corev1.SecurityContext{
        ReadOnlyRootFilesystem:   &readOnlyRoot,
        RunAsNonRoot:            &runAsNonRoot,
        RunAsUser:               &userID,
        AllowPrivilegeEscalation: &[]bool{false}[0],
        Capabilities: &corev1.Capabilities{
            Drop: []corev1.Capability{"ALL"},
        },
        SeccompProfile: &corev1.SeccompProfile{
            Type: corev1.SeccompProfileTypeRuntimeDefault,
        },
    }
}

// Pod Security Context for payment processing
func CreatePodSecurityContext() *corev1.PodSecurityContext {
    return &corev1.PodSecurityContext{
        RunAsNonRoot: &[]bool{true}[0],
        RunAsUser:    &[]int64{1000}[0],
        RunAsGroup:   &[]int64{3000}[0],
        FSGroup:      &[]int64{2000}[0],
        SeccompProfile: &corev1.SeccompProfile{
            Type: corev1.SeccompProfileTypeRuntimeDefault,
        },
    }
}
```

---

## 🌍 **Multi-Cluster Management**

### **Advanced Multi-Cluster Architecture**

```go
package multicluster

import (
    "context"
    "fmt"
    "sync"
    "time"

    "k8s.io/client-go/kubernetes"
    "k8s.io/client-go/tools/clientcmd"
    "sigs.k8s.io/controller-runtime/pkg/client"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    appsv1 "k8s.io/api/apps/v1"
    corev1 "k8s.io/api/core/v1"
)

// ClusterManager manages multiple Kubernetes clusters
type ClusterManager struct {
    clusters      map[string]*ClusterClient
    federationMgr *FederationManager
    syncManager   *CrossClusterSyncManager
    
    // Health monitoring
    healthChecker *ClusterHealthChecker
    
    // Load balancing
    loadBalancer *MultiClusterLoadBalancer
    
    // Disaster recovery
    disasterRecovery *DisasterRecoveryManager
    
    mu sync.RWMutex
}

type ClusterClient struct {
    Name         string
    Context      string
    Region       string
    Environment  string
    Client       kubernetes.Interface
    RuntimeClient client.Client
    
    // Cluster metadata
    Capabilities []string
    Resources    ClusterResources
    Status       ClusterStatus
    
    // Connection details
    Config       *rest.Config
    LastPing     time.Time
    Healthy      bool
}

type ClusterResources struct {
    CPUCapacity    int64  `json:"cpuCapacity"`
    MemoryCapacity int64  `json:"memoryCapacity"`
    StorageCapacity int64 `json:"storageCapacity"`
    NodeCount      int32  `json:"nodeCount"`
    
    // Available resources
    CPUAvailable    int64 `json:"cpuAvailable"`
    MemoryAvailable int64 `json:"memoryAvailable"`
    StorageAvailable int64 `json:"storageAvailable"`
}

type ClusterStatus struct {
    State        string    `json:"state"`
    LastUpdated  time.Time `json:"lastUpdated"`
    Version      string    `json:"version"`
    Healthy      bool      `json:"healthy"`
    
    // Performance metrics
    ResponseTime time.Duration `json:"responseTime"`
    ThroughputRPS float64     `json:"throughputRPS"`
    ErrorRate    float64      `json:"errorRate"`
    
    // Resource utilization
    CPUUtilization    float64 `json:"cpuUtilization"`
    MemoryUtilization float64 `json:"memoryUtilization"`
    StorageUtilization float64 `json:"storageUtilization"`
}

// Initialize multi-cluster manager
func NewClusterManager() *ClusterManager {
    return &ClusterManager{
        clusters:         make(map[string]*ClusterClient),
        federationMgr:    NewFederationManager(),
        syncManager:      NewCrossClusterSyncManager(),
        healthChecker:    NewClusterHealthChecker(),
        loadBalancer:     NewMultiClusterLoadBalancer(),
        disasterRecovery: NewDisasterRecoveryManager(),
    }
}

// Add cluster to management
func (cm *ClusterManager) AddCluster(ctx context.Context, config ClusterConfig) error {
    cm.mu.Lock()
    defer cm.mu.Unlock()

    // Load kubeconfig
    clientConfig, err := clientcmd.LoadFromFile(config.KubeconfigPath)
    if err != nil {
        return fmt.Errorf("failed to load kubeconfig: %w", err)
    }

    restConfig, err := clientcmd.BuildConfigFromFlags("", config.KubeconfigPath)
    if err != nil {
        return fmt.Errorf("failed to build rest config: %w", err)
    }

    // Create clients
    clientset, err := kubernetes.NewForConfig(restConfig)
    if err != nil {
        return fmt.Errorf("failed to create clientset: %w", err)
    }

    runtimeClient, err := client.New(restConfig, client.Options{})
    if err != nil {
        return fmt.Errorf("failed to create runtime client: %w", err)
    }

    // Create cluster client
    clusterClient := &ClusterClient{
        Name:          config.Name,
        Context:       config.Context,
        Region:        config.Region,
        Environment:   config.Environment,
        Client:        clientset,
        RuntimeClient: runtimeClient,
        Config:        restConfig,
        Capabilities:  config.Capabilities,
        LastPing:      time.Now(),
        Healthy:       true,
    }

    // Get cluster resources
    if err := cm.updateClusterResources(ctx, clusterClient); err != nil {
        return fmt.Errorf("failed to get cluster resources: %w", err)
    }

    cm.clusters[config.Name] = clusterClient

    // Start health monitoring
    go cm.healthChecker.MonitorCluster(ctx, clusterClient)

    return nil
}

// Cross-cluster deployment for payment processing
func (cm *ClusterManager) DeployPaymentProcessorAcrossClusters(ctx context.Context, spec PaymentProcessorSpec) error {
    // Select optimal clusters based on requirements
    selectedClusters, err := cm.selectClustersForDeployment(spec)
    if err != nil {
        return fmt.Errorf("failed to select clusters: %w", err)
    }

    var wg sync.WaitGroup
    errChan := make(chan error, len(selectedClusters))

    for _, cluster := range selectedClusters {
        wg.Add(1)
        go func(c *ClusterClient) {
            defer wg.Done()

            deployment := cm.createPaymentProcessorDeployment(spec, c)
            if err := cm.deployToCluster(ctx, c, deployment); err != nil {
                errChan <- fmt.Errorf("deployment failed on cluster %s: %w", c.Name, err)
                return
            }

            // Set up cross-cluster service discovery
            if err := cm.configureCrossClusterService(ctx, c, spec); err != nil {
                errChan <- fmt.Errorf("service configuration failed on cluster %s: %w", c.Name, err)
                return
            }
        }(cluster)
    }

    wg.Wait()
    close(errChan)

    // Check for any errors
    for err := range errChan {
        if err != nil {
            return err
        }
    }

    // Configure global load balancing
    return cm.loadBalancer.ConfigureGlobalLoadBalancing(ctx, selectedClusters, spec)
}

// Federation Manager for cross-cluster coordination
type FederationManager struct {
    federatedServices map[string]*FederatedService
    policies         []FederationPolicy
    mu               sync.RWMutex
}

type FederatedService struct {
    Name        string                    `json:"name"`
    Clusters    []string                  `json:"clusters"`
    Endpoints   map[string][]Endpoint     `json:"endpoints"`
    Policy      *FederationPolicy         `json:"policy"`
    Status      FederatedServiceStatus    `json:"status"`
}

type FederationPolicy struct {
    LoadBalancing  LoadBalancingPolicy  `json:"loadBalancing"`
    FailoverPolicy FailoverPolicy       `json:"failoverPolicy"`
    Locality       LocalityPolicy       `json:"locality"`
    Security       SecurityPolicy       `json:"security"`
}

// Configure federated payment service
func (fm *FederationManager) CreateFederatedPaymentService(ctx context.Context, clusters []*ClusterClient) error {
    service := &FederatedService{
        Name:     "payment-processor",
        Clusters: make([]string, len(clusters)),
        Endpoints: make(map[string][]Endpoint),
        Policy: &FederationPolicy{
            LoadBalancing: LoadBalancingPolicy{
                Algorithm: "WEIGHTED_ROUND_ROBIN",
                Weights:   map[string]int{},
            },
            FailoverPolicy: FailoverPolicy{
                Enabled:        true,
                MaxRetries:     3,
                TimeoutSeconds: 30,
                BackoffPolicy:  "EXPONENTIAL",
            },
            Locality: LocalityPolicy{
                Enabled:           true,
                PreferLocalZone:   true,
                CrossZoneFailover: true,
            },
        },
    }

    for i, cluster := range clusters {
        service.Clusters[i] = cluster.Name
        
        // Discover service endpoints in each cluster
        endpoints, err := fm.discoverServiceEndpoints(ctx, cluster, "payment-processor")
        if err != nil {
            return fmt.Errorf("failed to discover endpoints in cluster %s: %w", cluster.Name, err)
        }
        
        service.Endpoints[cluster.Name] = endpoints
        service.Policy.LoadBalancing.Weights[cluster.Name] = fm.calculateClusterWeight(cluster)
    }

    fm.mu.Lock()
    fm.federatedServices["payment-processor"] = service
    fm.mu.Unlock()

    // Configure cross-cluster service mesh
    return fm.configureCrossClusterServiceMesh(ctx, service)
}

// Cross-cluster synchronization manager
type CrossClusterSyncManager struct {
    syncPolicies map[string]SyncPolicy
    secretSync   *SecretSynchronizer
    configSync   *ConfigSynchronizer
    mu           sync.RWMutex
}

type SyncPolicy struct {
    ResourceType string            `json:"resourceType"`
    Namespaces   []string          `json:"namespaces"`
    Selector     metav1.LabelSelector `json:"selector"`
    Clusters     []string          `json:"clusters"`
    Strategy     SyncStrategy      `json:"strategy"`
}

type SyncStrategy struct {
    Mode           string        `json:"mode"` // "PUSH", "PULL", "BIDIRECTIONAL"
    Interval       time.Duration `json:"interval"`
    ConflictPolicy string        `json:"conflictPolicy"` // "SOURCE_WINS", "NEWEST_WINS", "MANUAL"
}

// Synchronize payment processor secrets across clusters
func (csm *CrossClusterSyncManager) SyncPaymentSecrets(ctx context.Context, clusters []*ClusterClient) error {
    policy := SyncPolicy{
        ResourceType: "Secret",
        Namespaces:   []string{"payment-system", "banking-integrations"},
        Selector: metav1.LabelSelector{
            MatchLabels: map[string]string{
                "type": "payment-credentials",
            },
        },
        Strategy: SyncStrategy{
            Mode:           "PUSH",
            Interval:       5 * time.Minute,
            ConflictPolicy: "SOURCE_WINS",
        },
    }

    // Get source cluster (primary)
    sourceCluster := clusters[0] // Assume first cluster is primary

    // Get secrets from source
    secrets, err := csm.getSecretsFromCluster(ctx, sourceCluster, policy)
    if err != nil {
        return fmt.Errorf("failed to get secrets from source cluster: %w", err)
    }

    // Sync to target clusters
    for _, targetCluster := range clusters[1:] {
        if err := csm.syncSecretsToCluster(ctx, targetCluster, secrets, policy); err != nil {
            return fmt.Errorf("failed to sync secrets to cluster %s: %w", targetCluster.Name, err)
        }
    }

    return nil
}

// Disaster Recovery Manager
type DisasterRecoveryManager struct {
    backupManager    *BackupManager
    recoveryPlans    map[string]*RecoveryPlan
    failoverManager  *FailoverManager
    mu               sync.RWMutex
}

type RecoveryPlan struct {
    Name              string              `json:"name"`
    PrimaryCluster    string              `json:"primaryCluster"`
    BackupClusters    []string            `json:"backupClusters"`
    Services          []string            `json:"services"`
    RecoveryObjective RecoveryObjective   `json:"recoveryObjective"`
    Steps             []RecoveryStep      `json:"steps"`
}

type RecoveryObjective struct {
    RTO time.Duration `json:"rto"` // Recovery Time Objective
    RPO time.Duration `json:"rpo"` // Recovery Point Objective
}

// Execute disaster recovery for payment processing
func (drm *DisasterRecoveryManager) ExecutePaymentSystemRecovery(ctx context.Context, failedCluster string) error {
    plan, exists := drm.recoveryPlans["payment-system"]
    if !exists {
        return fmt.Errorf("no recovery plan found for payment-system")
    }

    // Start recovery timer
    recoveryStart := time.Now()

    // Step 1: Drain traffic from failed cluster
    if err := drm.failoverManager.DrainTraffic(ctx, failedCluster); err != nil {
        return fmt.Errorf("failed to drain traffic: %w", err)
    }

    // Step 2: Restore data from backups
    for _, service := range plan.Services {
        if err := drm.backupManager.RestoreService(ctx, service, plan.BackupClusters[0]); err != nil {
            return fmt.Errorf("failed to restore service %s: %w", service, err)
        }
    }

    // Step 3: Redirect traffic to backup clusters
    for _, backupCluster := range plan.BackupClusters {
        if err := drm.failoverManager.RedirectTraffic(ctx, failedCluster, backupCluster); err != nil {
            return fmt.Errorf("failed to redirect traffic to %s: %w", backupCluster, err)
        }
    }

    // Verify RTO compliance
    if time.Since(recoveryStart) > plan.RecoveryObjective.RTO {
        return fmt.Errorf("recovery exceeded RTO of %v", plan.RecoveryObjective.RTO)
    }

    return nil
}
```

---

## 🚀 **GitOps & Advanced Deployment Strategies**

### **Advanced GitOps Implementation**

```go
package gitops

import (
    "context"
    "fmt"
    "os"
    "path/filepath"
    "time"

    "github.com/go-git/go-git/v5"
    "github.com/go-git/go-git/v5/plumbing/object"
    argocd "github.com/argoproj/argo-cd/v2/pkg/client/clientset/versioned"
    argov1alpha1 "github.com/argoproj/argo-cd/v2/pkg/apis/application/v1alpha1"
    fluxv2 "github.com/fluxcd/flux2/api/v1beta1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/client-go/kubernetes"
    "sigs.k8s.io/controller-runtime/pkg/client"
)

// GitOpsManager manages GitOps workflows and deployments
type GitOpsManager struct {
    k8sClient   kubernetes.Interface
    argoClient  argocd.Interface
    fluxClient  client.Client
    
    // Repository management
    repoManager     *RepositoryManager
    
    // Application management
    appManager      *ApplicationManager
    
    // Deployment strategies
    deploymentMgr   *DeploymentStrategyManager
    
    // Progressive delivery
    progressiveMgr  *ProgressiveDeliveryManager
}

// Repository Manager for Git operations
type RepositoryManager struct {
    repositories map[string]*GitRepository
    webhookMgr   *WebhookManager
}

type GitRepository struct {
    Name       string         `json:"name"`
    URL        string         `json:"url"`
    Branch     string         `json:"branch"`
    Path       string         `json:"path"`
    Repository *git.Repository
    
    // Authentication
    Auth       GitAuth        `json:"auth"`
    
    // Sync configuration
    SyncPolicy SyncPolicy     `json:"syncPolicy"`
    
    // Webhook configuration
    Webhooks   []WebhookConfig `json:"webhooks"`
}

type GitAuth struct {
    Type         string `json:"type"` // "ssh", "token", "basic"
    Username     string `json:"username,omitempty"`
    Password     string `json:"password,omitempty"`
    SSHKey       string `json:"sshKey,omitempty"`
    Token        string `json:"token,omitempty"`
}

// Application Manager for GitOps applications
type ApplicationManager struct {
    applications map[string]*GitOpsApplication
}

type GitOpsApplication struct {
    Name        string                    `json:"name"`
    Namespace   string                    `json:"namespace"`
    Source      ApplicationSource         `json:"source"`
    Destination ApplicationDestination    `json:"destination"`
    SyncPolicy  ApplicationSyncPolicy     `json:"syncPolicy"`
    Health      ApplicationHealth         `json:"health"`
    Status      ApplicationStatus         `json:"status"`
    
    // Progressive delivery
    Rollout     *RolloutConfiguration     `json:"rollout,omitempty"`
}

type ApplicationSource struct {
    RepoURL        string            `json:"repoURL"`
    Path           string            `json:"path"`
    TargetRevision string            `json:"targetRevision"`
    Helm           *HelmSource       `json:"helm,omitempty"`
    Kustomize      *KustomizeSource  `json:"kustomize,omitempty"`
}

// Create payment processor GitOps application
func (am *ApplicationManager) CreatePaymentProcessorApp(ctx context.Context, argoClient argocd.Interface) error {
    app := &argov1alpha1.Application{
        ObjectMeta: metav1.ObjectMeta{
            Name:      "payment-processor",
            Namespace: "argocd",
            Labels: map[string]string{
                "environment": "production",
                "component":   "payment-system",
            },
            Annotations: map[string]string{
                "argocd.argoproj.io/sync-wave": "1",
            },
        },
        Spec: argov1alpha1.ApplicationSpec{
            Project: "payment-platform",
            Source: argov1alpha1.ApplicationSource{
                RepoURL:        "https://github.com/razorpay/payment-infrastructure",
                Path:           "applications/payment-processor",
                TargetRevision: "HEAD",
                Helm: &argov1alpha1.ApplicationSourceHelm{
                    ValueFiles: []string{
                        "values-production.yaml",
                        "values-secrets.yaml",
                    },
                    Parameters: []argov1alpha1.HelmParameter{
                        {
                            Name:  "image.tag",
                            Value: "v1.2.3",
                        },
                        {
                            Name:  "replicas",
                            Value: "5",
                        },
                        {
                            Name:  "resources.limits.memory",
                            Value: "2Gi",
                        },
                    },
                },
            },
            Destination: argov1alpha1.ApplicationDestination{
                Server:    "https://kubernetes.default.svc",
                Namespace: "payment-system",
            },
            SyncPolicy: &argov1alpha1.SyncPolicy{
                Automated: &argov1alpha1.SyncPolicyAutomated{
                    Prune:    true,
                    SelfHeal: true,
                },
                SyncOptions: []string{
                    "CreateNamespace=true",
                    "PrunePropagationPolicy=foreground",
                    "PruneLast=true",
                },
                Retry: &argov1alpha1.RetryStrategy{
                    Limit: 3,
                    Backoff: &argov1alpha1.Backoff{
                        Duration:    "30s",
                        Factor:      int64(2),
                        MaxDuration: "3m",
                    },
                },
            },
            RevisionHistoryLimit: int64(10),
        },
    }

    _, err := argoClient.ArgoprojV1alpha1().Applications("argocd").Create(ctx, app, metav1.CreateOptions{})
    return err
}

// Deployment Strategy Manager
type DeploymentStrategyManager struct {
    strategies map[string]DeploymentStrategy
}

type DeploymentStrategy interface {
    Execute(ctx context.Context, app *GitOpsApplication) error
    Validate(ctx context.Context, app *GitOpsApplication) error
    Rollback(ctx context.Context, app *GitOpsApplication) error
}

// Blue-Green Deployment Strategy
type BlueGreenStrategy struct {
    client         kubernetes.Interface
    trafficManager *TrafficManager
}

func (bgs *BlueGreenStrategy) Execute(ctx context.Context, app *GitOpsApplication) error {
    // Step 1: Deploy new version to "green" environment
    greenDeployment := bgs.createGreenDeployment(app)
    if err := bgs.deployGreenEnvironment(ctx, greenDeployment); err != nil {
        return fmt.Errorf("failed to deploy green environment: %w", err)
    }

    // Step 2: Run health checks and validation
    if err := bgs.validateGreenEnvironment(ctx, app); err != nil {
        return fmt.Errorf("green environment validation failed: %w", err)
    }

    // Step 3: Switch traffic from blue to green
    if err := bgs.trafficManager.SwitchTrafficToGreen(ctx, app.Name); err != nil {
        return fmt.Errorf("failed to switch traffic: %w", err)
    }

    // Step 4: Monitor for issues
    if err := bgs.monitorDeployment(ctx, app, 5*time.Minute); err != nil {
        // Rollback on failure
        if rollbackErr := bgs.trafficManager.SwitchTrafficToBlue(ctx, app.Name); rollbackErr != nil {
            return fmt.Errorf("deployment failed and rollback failed: %v, %v", err, rollbackErr)
        }
        return fmt.Errorf("deployment failed, rolled back: %w", err)
    }

    // Step 5: Cleanup old blue environment
    return bgs.cleanupBlueEnvironment(ctx, app)
}

// Canary Deployment Strategy
type CanaryStrategy struct {
    client          kubernetes.Interface
    trafficSplitter *TrafficSplitter
    metricCollector *MetricCollector
}

func (cs *CanaryStrategy) Execute(ctx context.Context, app *GitOpsApplication) error {
    phases := []CanaryPhase{
        {TrafficPercent: 5, Duration: 2 * time.Minute},
        {TrafficPercent: 10, Duration: 5 * time.Minute},
        {TrafficPercent: 25, Duration: 10 * time.Minute},
        {TrafficPercent: 50, Duration: 15 * time.Minute},
        {TrafficPercent: 100, Duration: 0},
    }

    for i, phase := range phases {
        // Deploy canary with traffic percentage
        if err := cs.trafficSplitter.SetCanaryTraffic(ctx, app.Name, phase.TrafficPercent); err != nil {
            return fmt.Errorf("failed to set canary traffic to %d%%: %w", phase.TrafficPercent, err)
        }

        // Wait for phase duration
        if phase.Duration > 0 {
            time.Sleep(phase.Duration)
        }

        // Analyze metrics
        metrics, err := cs.metricCollector.GetCanaryMetrics(ctx, app.Name, phase.Duration)
        if err != nil {
            return fmt.Errorf("failed to collect metrics for phase %d: %w", i+1, err)
        }

        // Evaluate success criteria
        if !cs.evaluateSuccessCriteria(metrics, app.Rollout.SuccessCriteria) {
            // Rollback canary
            if err := cs.trafficSplitter.SetCanaryTraffic(ctx, app.Name, 0); err != nil {
                return fmt.Errorf("canary failed and rollback failed: %w", err)
            }
            return fmt.Errorf("canary deployment failed success criteria at phase %d", i+1)
        }
    }

    return nil
}

// Progressive Delivery Manager
type ProgressiveDeliveryManager struct {
    rolloutManager  *RolloutManager
    experimentMgr   *ExperimentManager
    analysisRunner  *AnalysisRunner
}

type RolloutConfiguration struct {
    Strategy        string                 `json:"strategy"`
    Steps           []RolloutStep          `json:"steps"`
    SuccessCriteria []SuccessCriterion     `json:"successCriteria"`
    FailurePolicy   FailurePolicy          `json:"failurePolicy"`
    Analysis        *AnalysisConfiguration `json:"analysis,omitempty"`
}

type RolloutStep struct {
    Weight          int32         `json:"weight,omitempty"`
    Pause           *PauseStep    `json:"pause,omitempty"`
    SetWeight       *int32        `json:"setWeight,omitempty"`
    Experiment      *Experiment   `json:"experiment,omitempty"`
    Analysis        *AnalysisStep `json:"analysis,omitempty"`
}

// Advanced rollout with analysis
func (pdm *ProgressiveDeliveryManager) ExecuteProgressiveRollout(ctx context.Context, app *GitOpsApplication) error {
    rollout := app.Rollout
    if rollout == nil {
        return fmt.Errorf("no rollout configuration provided")
    }

    for i, step := range rollout.Steps {
        switch {
        case step.SetWeight != nil:
            if err := pdm.rolloutManager.SetTrafficWeight(ctx, app.Name, *step.SetWeight); err != nil {
                return fmt.Errorf("failed to set traffic weight at step %d: %w", i+1, err)
            }

        case step.Pause != nil:
            if step.Pause.Duration != nil {
                time.Sleep(*step.Pause.Duration)
            } else {
                // Manual approval required
                if err := pdm.waitForManualApproval(ctx, app.Name, step.Pause); err != nil {
                    return fmt.Errorf("manual approval failed at step %d: %w", i+1, err)
                }
            }

        case step.Experiment != nil:
            result, err := pdm.experimentMgr.RunExperiment(ctx, step.Experiment)
            if err != nil {
                return fmt.Errorf("experiment failed at step %d: %w", i+1, err)
            }
            if !result.Successful {
                return fmt.Errorf("experiment unsuccessful at step %d: %s", i+1, result.Reason)
            }

        case step.Analysis != nil:
            result, err := pdm.analysisRunner.RunAnalysis(ctx, step.Analysis)
            if err != nil {
                return fmt.Errorf("analysis failed at step %d: %w", i+1, err)
            }
            if !result.Successful {
                return fmt.Errorf("analysis unsuccessful at step %d: %s", i+1, result.Reason)
            }
        }

        // Check overall success criteria after each step
        if !pdm.evaluateOverallCriteria(ctx, app, rollout.SuccessCriteria) {
            return pdm.executeFailurePolicy(ctx, app, rollout.FailurePolicy, i+1)
        }
    }

    return nil
}

// Webhook Manager for Git events
type WebhookManager struct {
    handlers map[string]WebhookHandler
    server   *http.Server
}

type WebhookHandler interface {
    Handle(ctx context.Context, event WebhookEvent) error
}

// Payment processor webhook handler
type PaymentProcessorWebhookHandler struct {
    appManager *ApplicationManager
    syncTrigger *SyncTrigger
}

func (pph *PaymentProcessorWebhookHandler) Handle(ctx context.Context, event WebhookEvent) error {
    if event.Repository != "payment-infrastructure" {
        return nil // Not our repository
    }

    switch event.Type {
    case "push":
        // Trigger sync for payment processor applications
        apps := []string{"payment-processor", "payment-gateway", "fraud-detection"}
        for _, appName := range apps {
            if err := pph.syncTrigger.TriggerSync(ctx, appName); err != nil {
                return fmt.Errorf("failed to trigger sync for %s: %w", appName, err)
            }
        }

    case "pull_request":
        // Create preview environment
        if err := pph.createPreviewEnvironment(ctx, event.PullRequest); err != nil {
            return fmt.Errorf("failed to create preview environment: %w", err)
        }

    case "release":
        // Trigger production deployment
        if err := pph.triggerProductionDeployment(ctx, event.Release); err != nil {
            return fmt.Errorf("failed to trigger production deployment: %w", err)
        }
    }

    return nil
}

// Environment promotion pipeline
func (gm *GitOpsManager) CreatePromotionPipeline(ctx context.Context, service string) error {
    environments := []Environment{
        {Name: "dev", Cluster: "dev-cluster", AutoPromote: true},
        {Name: "staging", Cluster: "staging-cluster", AutoPromote: false},
        {Name: "prod", Cluster: "prod-cluster", AutoPromote: false},
    }

    for i, env := range environments {
        app := &GitOpsApplication{
            Name:      fmt.Sprintf("%s-%s", service, env.Name),
            Namespace: "argocd",
            Source: ApplicationSource{
                RepoURL:        "https://github.com/razorpay/payment-infrastructure",
                Path:           fmt.Sprintf("environments/%s/%s", env.Name, service),
                TargetRevision: "HEAD",
            },
            Destination: ApplicationDestination{
                Server:    env.Cluster,
                Namespace: fmt.Sprintf("%s-%s", service, env.Name),
            },
            SyncPolicy: ApplicationSyncPolicy{
                Automated: env.AutoPromote,
                SyncOptions: []string{
                    "CreateNamespace=true",
                    "PrunePropagationPolicy=foreground",
                },
            },
        }

        // Configure promotion triggers
        if i > 0 {
            app.PromotionTrigger = &PromotionTrigger{
                SourceEnvironment: environments[i-1].Name,
                Conditions: []PromotionCondition{
                    {Type: "HealthCheck", Status: "Healthy"},
                    {Type: "TestSuite", Status: "Passed"},
                    {Type: "SecurityScan", Status: "Passed"},
                },
                ApprovalRequired: !env.AutoPromote,
            }
        }

        if err := gm.appManager.CreateApplication(ctx, app); err != nil {
            return fmt.Errorf("failed to create application for %s environment: %w", env.Name, err)
        }
    }

    return nil
}
```

---

## 📊 **Advanced Scheduling & Resource Management**

### **Sophisticated Scheduling Strategies**

```go
package scheduling

import (
    "context"
    "fmt"
    "time"

    corev1 "k8s.io/api/core/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/fields"
    "k8s.io/apimachinery/pkg/labels"
    "k8s.io/client-go/kubernetes"
    schedulingv1 "k8s.io/api/scheduling/v1"
    policyv1 "k8s.io/api/policy/v1"
)

// AdvancedScheduler manages complex scheduling requirements
type AdvancedScheduler struct {
    clientset           kubernetes.Interface
    priorityManager     *PriorityClassManager
    affinityManager     *AffinityManager
    taintManager        *TaintTolerationManager
    topologyManager     *TopologyManager
    resourceManager     *ResourceManager
}

// Priority Class Manager for workload prioritization
type PriorityClassManager struct {
    clientset kubernetes.Interface
}

// Create priority classes for payment system components
func (pcm *PriorityClassManager) CreatePaymentSystemPriorities(ctx context.Context) error {
    priorities := []struct {
        name        string
        value       int32
        description string
        global      bool
    }{
        {
            name:        "payment-critical",
            value:       1000000,
            description: "Critical payment processing workloads",
            global:      true,
        },
        {
            name:        "payment-high",
            value:       100000,
            description: "High priority payment workloads",
            global:      true,
        },
        {
            name:        "payment-normal",
            value:       1000,
            description: "Normal payment processing workloads",
            global:      true,
        },
        {
            name:        "payment-low",
            value:       100,
            description: "Low priority batch processing",
            global:      true,
        },
    }

    for _, p := range priorities {
        pc := &schedulingv1.PriorityClass{
            ObjectMeta: metav1.ObjectMeta{
                Name: p.name,
            },
            Value:            p.value,
            GlobalDefault:    false,
            Description:      p.description,
            PreemptionPolicy: &[]corev1.PreemptionPolicy{corev1.PreemptLowerPriority}[0],
        }

        if _, err := pcm.clientset.SchedulingV1().PriorityClasses().Create(ctx, pc, metav1.CreateOptions{}); err != nil {
            return fmt.Errorf("failed to create priority class %s: %w", p.name, err)
        }
    }

    return nil
}

// Affinity Manager for advanced pod placement
type AffinityManager struct {
    clientset kubernetes.Interface
}

// Create complex affinity rules for payment processing
func (am *AffinityManager) CreatePaymentProcessorAffinity() *corev1.Affinity {
    return &corev1.Affinity{
        NodeAffinity: &corev1.NodeAffinity{
            RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
                NodeSelectorTerms: []corev1.NodeSelectorTerm{
                    {
                        MatchExpressions: []corev1.NodeSelectorRequirement{
                            {
                                Key:      "node.kubernetes.io/instance-type",
                                Operator: corev1.NodeSelectorOpIn,
                                Values:   []string{"c5.xlarge", "c5.2xlarge", "c5.4xlarge"},
                            },
                            {
                                Key:      "kubernetes.io/arch",
                                Operator: corev1.NodeSelectorOpIn,
                                Values:   []string{"amd64"},
                            },
                            {
                                Key:      "failure-domain.beta.kubernetes.io/zone",
                                Operator: corev1.NodeSelectorOpIn,
                                Values:   []string{"us-east-1a", "us-east-1b", "us-east-1c"},
                            },
                        },
                    },
                },
            },
            PreferredDuringSchedulingIgnoredDuringExecution: []corev1.PreferredSchedulingTerm{
                {
                    Weight: 100,
                    Preference: corev1.NodeSelectorTerm{
                        MatchExpressions: []corev1.NodeSelectorRequirement{
                            {
                                Key:      "node-type",
                                Operator: corev1.NodeSelectorOpIn,
                                Values:   []string{"payment-optimized"},
                            },
                        },
                    },
                },
                {
                    Weight: 50,
                    Preference: corev1.NodeSelectorTerm{
                        MatchExpressions: []corev1.NodeSelectorRequirement{
                            {
                                Key:      "spot-instance",
                                Operator: corev1.NodeSelectorOpNotIn,
                                Values:   []string{"true"},
                            },
                        },
                    },
                },
            },
        },
        PodAffinity: &corev1.PodAffinity{
            PreferredDuringSchedulingIgnoredDuringExecution: []corev1.WeightedPodAffinityTerm{
                {
                    Weight: 100,
                    PodAffinityTerm: corev1.PodAffinityTerm{
                        LabelSelector: &metav1.LabelSelector{
                            MatchLabels: map[string]string{
                                "component": "database",
                                "tier":      "cache",
                            },
                        },
                        TopologyKey: "kubernetes.io/hostname",
                    },
                },
            },
        },
        PodAntiAffinity: &corev1.PodAntiAffinity{
            RequiredDuringSchedulingIgnoredDuringExecution: []corev1.PodAffinityTerm{
                {
                    LabelSelector: &metav1.LabelSelector{
                        MatchLabels: map[string]string{
                            "app": "payment-processor",
                        },
                    },
                    TopologyKey: "kubernetes.io/hostname",
                },
            },
            PreferredDuringSchedulingIgnoredDuringExecution: []corev1.WeightedPodAffinityTerm{
                {
                    Weight: 50,
                    PodAffinityTerm: corev1.PodAffinityTerm{
                        LabelSelector: &metav1.LabelSelector{
                            MatchLabels: map[string]string{
                                "app": "fraud-detection",
                            },
                        },
                        TopologyKey: "failure-domain.beta.kubernetes.io/zone",
                    },
                },
            },
        },
    }
}

// Topology Spread Constraints for balanced distribution
func (am *AffinityManager) CreateTopologySpreadConstraints() []corev1.TopologySpreadConstraint {
    return []corev1.TopologySpreadConstraint{
        {
            MaxSkew:           1,
            TopologyKey:       "failure-domain.beta.kubernetes.io/zone",
            WhenUnsatisfiable: corev1.DoNotSchedule,
            LabelSelector: &metav1.LabelSelector{
                MatchLabels: map[string]string{
                    "app": "payment-processor",
                },
            },
        },
        {
            MaxSkew:           2,
            TopologyKey:       "kubernetes.io/hostname",
            WhenUnsatisfiable: corev1.ScheduleAnyway,
            LabelSelector: &metav1.LabelSelector{
                MatchLabels: map[string]string{
                    "component": "payment-system",
                },
            },
        },
    }
}

// Taint and Toleration Manager
type TaintTolerationManager struct {
    clientset kubernetes.Interface
}

// Apply payment-specific taints to nodes
func (ttm *TaintTolerationManager) TaintPaymentNodes(ctx context.Context, nodeSelector map[string]string) error {
    nodes, err := ttm.clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{
        LabelSelector: labels.SelectorFromSet(nodeSelector).String(),
    })
    if err != nil {
        return fmt.Errorf("failed to list nodes: %w", err)
    }

    paymentTaint := corev1.Taint{
        Key:    "workload-type",
        Value:  "payment-processing",
        Effect: corev1.TaintEffectNoSchedule,
    }

    for _, node := range nodes.Items {
        node.Spec.Taints = append(node.Spec.Taints, paymentTaint)
        if _, err := ttm.clientset.CoreV1().Nodes().Update(ctx, &node, metav1.UpdateOptions{}); err != nil {
            return fmt.Errorf("failed to taint node %s: %w", node.Name, err)
        }
    }

    return nil
}

// Create tolerations for payment workloads
func (ttm *TaintTolerationManager) CreatePaymentTolerations() []corev1.Toleration {
    return []corev1.Toleration{
        {
            Key:      "workload-type",
            Operator: corev1.TolerationOpEqual,
            Value:    "payment-processing",
            Effect:   corev1.TaintEffectNoSchedule,
        },
        {
            Key:      "node.kubernetes.io/memory-pressure",
            Operator: corev1.TolerationOpExists,
            Effect:   corev1.TaintEffectNoSchedule,
            TolerationSeconds: &[]int64{300}[0], // Tolerate for 5 minutes
        },
        {
            Key:      "spot-instance",
            Operator: corev1.TolerationOpEqual,
            Value:    "true",
            Effect:   corev1.TaintEffectNoSchedule,
        },
    }
}

// Resource Manager for advanced resource allocation
type ResourceManager struct {
    clientset kubernetes.Interface
}

// Create resource quotas for payment namespaces
func (rm *ResourceManager) CreatePaymentResourceQuotas(ctx context.Context, namespace string) error {
    resourceQuota := &corev1.ResourceQuota{
        ObjectMeta: metav1.ObjectMeta{
            Name:      "payment-system-quota",
            Namespace: namespace,
        },
        Spec: corev1.ResourceQuotaSpec{
            Hard: corev1.ResourceList{
                // Compute resources
                corev1.ResourceRequestsCPU:    resource.MustParse("20"),
                corev1.ResourceLimitsCPU:      resource.MustParse("40"),
                corev1.ResourceRequestsMemory: resource.MustParse("40Gi"),
                corev1.ResourceLimitsMemory:   resource.MustParse("80Gi"),
                
                // Storage resources
                corev1.ResourceRequestsStorage: resource.MustParse("1Ti"),
                
                // Object count limits
                corev1.ResourcePods:                   resource.MustParse("50"),
                corev1.ResourceServices:               resource.MustParse("20"),
                corev1.ResourceConfigMaps:             resource.MustParse("30"),
                corev1.ResourceSecrets:                resource.MustParse("20"),
                corev1.ResourcePersistentVolumeClaims: resource.MustParse("10"),
                
                // Extended resources
                "nvidia.com/gpu": resource.MustParse("4"),
            },
            ScopeSelector: &corev1.ScopeSelector{
                MatchExpressions: []corev1.ScopedResourceSelectorRequirement{
                    {
                        ScopeName: corev1.ResourceQuotaScopePriorityClass,
                        Operator:  corev1.ScopeSelectorOpIn,
                        Values:    []string{"payment-critical", "payment-high"},
                    },
                },
            },
        },
    }

    if _, err := rm.clientset.CoreV1().ResourceQuotas(namespace).Create(ctx, resourceQuota, metav1.CreateOptions{}); err != nil {
        return fmt.Errorf("failed to create resource quota: %w", err)
    }

    return nil
}

// Create Pod Disruption Budget for high availability
func (rm *ResourceManager) CreatePodDisruptionBudget(ctx context.Context, namespace string, appName string) error {
    pdb := &policyv1.PodDisruptionBudget{
        ObjectMeta: metav1.ObjectMeta{
            Name:      fmt.Sprintf("%s-pdb", appName),
            Namespace: namespace,
        },
        Spec: policyv1.PodDisruptionBudgetSpec{
            MinAvailable: &intstr.FromString("50%"),
            Selector: &metav1.LabelSelector{
                MatchLabels: map[string]string{
                    "app": appName,
                },
            },
        },
    }

    if _, err := rm.clientset.PolicyV1().PodDisruptionBudgets(namespace).Create(ctx, pdb, metav1.CreateOptions{}); err != nil {
        return fmt.Errorf("failed to create pod disruption budget: %w", err)
    }

    return nil
}
```

---

## 📈 **Advanced Observability & Monitoring**

### **Comprehensive Monitoring Stack**

```go
package observability

import (
    "context"
    "fmt"
    "time"

    promv1 "github.com/prometheus-operator/prometheus-operator/pkg/apis/monitoring/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/util/intstr"
    "sigs.k8s.io/controller-runtime/pkg/client"
    appsv1 "k8s.io/api/apps/v1"
    corev1 "k8s.io/api/core/v1"
)

// ObservabilityManager manages monitoring and observability
type ObservabilityManager struct {
    client              client.Client
    prometheusManager   *PrometheusManager
    grafanaManager      *GrafanaManager
    jaegerManager       *JaegerManager
    elasticManager      *ElasticSearchManager
    alertManager        *AlertManager
}

// Prometheus Manager for metrics collection
type PrometheusManager struct {
    client client.Client
}

// Create ServiceMonitor for payment processor
func (pm *PrometheusManager) CreatePaymentServiceMonitor(ctx context.Context, namespace string) error {
    serviceMonitor := &promv1.ServiceMonitor{
        ObjectMeta: metav1.ObjectMeta{
            Name:      "payment-processor-metrics",
            Namespace: namespace,
            Labels: map[string]string{
                "app":       "payment-processor",
                "component": "monitoring",
            },
        },
        Spec: promv1.ServiceMonitorSpec{
            Selector: metav1.LabelSelector{
                MatchLabels: map[string]string{
                    "app": "payment-processor",
                },
            },
            Endpoints: []promv1.Endpoint{
                {
                    Port:     "metrics",
                    Path:     "/metrics",
                    Interval: "30s",
                    Scheme:   "http",
                    Params: map[string][]string{
                        "format": {"prometheus"},
                    },
                    RelabelConfigs: []*promv1.RelabelConfig{
                        {
                            SourceLabels: []promv1.LabelName{"__meta_kubernetes_pod_name"},
                            TargetLabel:  "pod",
                            Action:       "replace",
                        },
                        {
                            SourceLabels: []promv1.LabelName{"__meta_kubernetes_pod_node_name"},
                            TargetLabel:  "node",
                            Action:       "replace",
                        },
                    },
                    MetricRelabelConfigs: []*promv1.RelabelConfig{
                        {
                            SourceLabels: []promv1.LabelName{"__name__"},
                            Regex:        "payment_processor_.*",
                            Action:       "keep",
                        },
                    },
                },
                {
                    Port:     "admin",
                    Path:     "/admin/metrics",
                    Interval: "60s",
                    Scheme:   "https",
                    TLSConfig: &promv1.TLSConfig{
                        SafeTLSConfig: promv1.SafeTLSConfig{
                            InsecureSkipVerify: true,
                        },
                    },
                },
            },
            NamespaceSelector: promv1.NamespaceSelector{
                MatchNames: []string{namespace},
            },
            PodTargetLabels: []string{
                "version",
                "environment",
                "tier",
            },
        },
    }

    return pm.client.Create(ctx, serviceMonitor)
}

// Create PrometheusRule for payment alerts
func (pm *PrometheusManager) CreatePaymentPrometheusRules(ctx context.Context, namespace string) error {
    prometheusRule := &promv1.PrometheusRule{
        ObjectMeta: metav1.ObjectMeta{
            Name:      "payment-processor-rules",
            Namespace: namespace,
            Labels: map[string]string{
                "app":  "payment-processor",
                "role": "alert-rules",
            },
        },
        Spec: promv1.PrometheusRuleSpec{
            Groups: []promv1.RuleGroup{
                {
                    Name:     "payment-processor.rules",
                    Interval: "30s",
                    Rules: []promv1.Rule{
                        // Payment processing rate
                        {
                            Record: "payment:processing_rate",
                            Expr:   intstr.FromString("rate(payment_processor_transactions_total[5m])"),
                            Labels: map[string]string{
                                "service": "payment-processor",
                            },
                        },
                        // Payment success rate
                        {
                            Record: "payment:success_rate",
                            Expr: intstr.FromString(`
                                rate(payment_processor_transactions_total{status="success"}[5m]) /
                                rate(payment_processor_transactions_total[5m])
                            `),
                            Labels: map[string]string{
                                "service": "payment-processor",
                            },
                        },
                        // Payment latency percentiles
                        {
                            Record: "payment:latency_p99",
                            Expr:   intstr.FromString("histogram_quantile(0.99, rate(payment_processor_duration_seconds_bucket[5m]))"),
                        },
                        {
                            Record: "payment:latency_p95",
                            Expr:   intstr.FromString("histogram_quantile(0.95, rate(payment_processor_duration_seconds_bucket[5m]))"),
                        },
                        {
                            Record: "payment:latency_p50",
                            Expr:   intstr.FromString("histogram_quantile(0.50, rate(payment_processor_duration_seconds_bucket[5m]))"),
                        },
                    ],
                },
                {
                    Name:     "payment-processor.alerts",
                    Interval: "30s",
                    Rules: []promv1.Rule{
                        // High error rate alert
                        {
                            Alert: "PaymentProcessorHighErrorRate",
                            Expr:  intstr.FromString("payment:success_rate < 0.95"),
                            For:   "2m",
                            Annotations: map[string]string{
                                "summary":     "Payment processor has high error rate",
                                "description": "Payment success rate is {{ $value | humanizePercentage }} for more than 2 minutes",
                                "runbook_url": "https://runbooks.razorpay.com/payment-processor-high-error-rate",
                            },
                            Labels: map[string]string{
                                "severity": "critical",
                                "team":     "payments",
                                "service":  "payment-processor",
                            },
                        },
                        // High latency alert
                        {
                            Alert: "PaymentProcessorHighLatency",
                            Expr:  intstr.FromString("payment:latency_p95 > 2"),
                            For:   "5m",
                            Annotations: map[string]string{
                                "summary":     "Payment processor has high latency",
                                "description": "95th percentile latency is {{ $value }}s for more than 5 minutes",
                                "runbook_url": "https://runbooks.razorpay.com/payment-processor-high-latency",
                            },
                            Labels: map[string]string{
                                "severity": "warning",
                                "team":     "payments",
                                "service":  "payment-processor",
                            },
                        },
                        // Low throughput alert
                        {
                            Alert: "PaymentProcessorLowThroughput",
                            Expr:  intstr.FromString("payment:processing_rate < 100"),
                            For:   "10m",
                            Annotations: map[string]string{
                                "summary":     "Payment processor has low throughput",
                                "description": "Payment processing rate is {{ $value }} transactions/sec for more than 10 minutes",
                            },
                            Labels: map[string]string{
                                "severity": "warning",
                                "team":     "payments",
                                "service":  "payment-processor",
                            },
                        },
                        // Pod restart alert
                        {
                            Alert: "PaymentProcessorPodRestarting",
                            Expr:  intstr.FromString("rate(kube_pod_container_status_restarts_total{pod=~\"payment-processor-.*\"}[15m]) > 0"),
                            For:   "5m",
                            Annotations: map[string]string{
                                "summary":     "Payment processor pods are restarting frequently",
                                "description": "Pod {{ $labels.pod }} is restarting frequently",
                            },
                            Labels: map[string]string{
                                "severity": "warning",
                                "team":     "payments",
                                "service":  "payment-processor",
                            },
                        },
                    },
                },
            },
        },
    }

    return pm.client.Create(ctx, prometheusRule)
}

// Custom Grafana Dashboard for payment metrics
func (gm *GrafanaManager) CreatePaymentDashboard(ctx context.Context) (*Dashboard, error) {
    dashboard := &Dashboard{
        ID:          nil,
        Title:       "Payment Processing System",
        Description: "Comprehensive dashboard for payment processing monitoring",
        Tags:        []string{"payments", "fintech", "production"},
        Timezone:    "UTC",
        Refresh:     "30s",
        Time: TimeRange{
            From: "now-1h",
            To:   "now",
        },
        Panels: []Panel{
            {
                ID:    1,
                Title: "Payment Processing Rate",
                Type:  "graph",
                Targets: []Target{
                    {
                        Expr:         "payment:processing_rate",
                        LegendFormat: "Transactions/sec",
                        RefId:        "A",
                    },
                },
                YAxes: []YAxis{
                    {
                        Label: "Transactions/sec",
                        Min:   floatPtr(0),
                    },
                },
                GridPos: GridPos{X: 0, Y: 0, W: 12, H: 8},
            },
            {
                ID:    2,
                Title: "Success Rate",
                Type:  "singlestat",
                Targets: []Target{
                    {
                        Expr:         "payment:success_rate",
                        LegendFormat: "Success Rate",
                        RefId:        "A",
                    },
                },
                Format: "percentunit",
                Thresholds: []Threshold{
                    {Value: 0.95, ColorMode: "critical"},
                    {Value: 0.98, ColorMode: "warning"},
                },
                GridPos: GridPos{X: 12, Y: 0, W: 6, H: 4},
            },
            {
                ID:    3,
                Title: "Latency Percentiles",
                Type:  "graph",
                Targets: []Target{
                    {
                        Expr:         "payment:latency_p99",
                        LegendFormat: "P99",
                        RefId:        "A",
                    },
                    {
                        Expr:         "payment:latency_p95",
                        LegendFormat: "P95",
                        RefId:        "B",
                    },
                    {
                        Expr:         "payment:latency_p50",
                        LegendFormat: "P50",
                        RefId:        "C",
                    },
                },
                YAxes: []YAxis{
                    {
                        Label: "Seconds",
                        Min:   floatPtr(0),
                    },
                },
                GridPos: GridPos{X: 18, Y: 0, W: 6, H: 8},
            },
            {
                ID:    4,
                Title: "Error Rate by Payment Method",
                Type:  "graph",
                Targets: []Target{
                    {
                        Expr:         "rate(payment_processor_errors_total[5m]) by (payment_method)",
                        LegendFormat: "{{ payment_method }}",
                        RefId:        "A",
                    },
                },
                GridPos: GridPos{X: 0, Y: 8, W: 12, H: 8},
            },
            {
                ID:    5,
                Title: "Resource Utilization",
                Type:  "graph",
                Targets: []Target{
                    {
                        Expr:         "rate(container_cpu_usage_seconds_total{pod=~\"payment-processor-.*\"}[5m]) * 100",
                        LegendFormat: "CPU %",
                        RefId:        "A",
                    },
                    {
                        Expr:         "container_memory_usage_bytes{pod=~\"payment-processor-.*\"} / 1024 / 1024",
                        LegendFormat: "Memory MB",
                        RefId:        "B",
                    },
                },
                GridPos: GridPos{X: 12, Y: 8, W: 12, H: 8},
            },
        },
        Templating: Templating{
            List: []Template{
                {
                    Name:  "namespace",
                    Type:  "query",
                    Query: "label_values(kube_pod_info, namespace)",
                    Multi: false,
                },
                {
                    Name:  "pod",
                    Type:  "query",
                    Query: "label_values(kube_pod_info{namespace=\"$namespace\"}, pod)",
                    Multi: true,
                },
            },
        },
    }

    return gm.createDashboard(ctx, dashboard)
}

// Distributed tracing with Jaeger
type JaegerManager struct {
    client client.Client
}

// Deploy Jaeger for distributed tracing
func (jm *JaegerManager) DeployJaeger(ctx context.Context, namespace string) error {
    jaegerDeployment := &appsv1.Deployment{
        ObjectMeta: metav1.ObjectMeta{
            Name:      "jaeger-all-in-one",
            Namespace: namespace,
            Labels: map[string]string{
                "app":       "jaeger",
                "component": "all-in-one",
            },
        },
        Spec: appsv1.DeploymentSpec{
            Replicas: int32Ptr(1),
            Selector: &metav1.LabelSelector{
                MatchLabels: map[string]string{
                    "app":       "jaeger",
                    "component": "all-in-one",
                },
            },
            Template: corev1.PodTemplateSpec{
                ObjectMeta: metav1.ObjectMeta{
                    Labels: map[string]string{
                        "app":       "jaeger",
                        "component": "all-in-one",
                    },
                },
                Spec: corev1.PodSpec{
                    Containers: []corev1.Container{
                        {
                            Name:  "jaeger",
                            Image: "jaegertracing/all-in-one:1.35",
                            Ports: []corev1.ContainerPort{
                                {ContainerPort: 16686, Name: "ui"},
                                {ContainerPort: 14268, Name: "collector"},
                                {ContainerPort: 6831, Name: "agent-compact", Protocol: corev1.ProtocolUDP},
                                {ContainerPort: 6832, Name: "agent-binary", Protocol: corev1.ProtocolUDP},
                                {ContainerPort: 5778, Name: "agent-config"},
                                {ContainerPort: 5775, Name: "zipkin-compact", Protocol: corev1.ProtocolUDP},
                            },
                            Env: []corev1.EnvVar{
                                {
                                    Name:  "COLLECTOR_ZIPKIN_HTTP_PORT",
                                    Value: "9411",
                                },
                                {
                                    Name:  "SPAN_STORAGE_TYPE",
                                    Value: "elasticsearch",
                                },
                                {
                                    Name:  "ES_SERVER_URLS",
                                    Value: "http://elasticsearch:9200",
                                },
                                {
                                    Name:  "ES_INDEX_PREFIX",
                                    Value: "payment-traces",
                                },
                            },
                            Resources: corev1.ResourceRequirements{
                                Requests: corev1.ResourceList{
                                    corev1.ResourceCPU:    resource.MustParse("100m"),
                                    corev1.ResourceMemory: resource.MustParse("256Mi"),
                                },
                                Limits: corev1.ResourceList{
                                    corev1.ResourceCPU:    resource.MustParse("500m"),
                                    corev1.ResourceMemory: resource.MustParse("512Mi"),
                                },
                            },
                        },
                    },
                },
            },
        },
    }

    return jm.client.Create(ctx, jaegerDeployment)
}
```

---

## 🎯 **Interview Questions & Practice Scenarios**

### **Advanced Kubernetes Operations - Technical Interview Questions**

#### **Custom Resources & Operators (Senior Level)**

**Q1: Design and implement a custom Kubernetes operator for managing payment processor deployments with automatic scaling based on transaction volume.**

*Expected Answer Points:*
- CRD design with proper validation schemas
- Operator controller implementation with reconciliation logic
- Integration with HPA based on custom metrics
- Status management and error handling
- Event recording and observability

**Q2: How would you handle operator upgrades in production without disrupting running workloads?**

*Expected Answer:*
- Blue-green deployment of operators
- Backward compatibility in CRD versions
- Leader election for HA operators
- Graceful shutdown and resource cleanup
- Rollback strategies

**Q3: Implement a multi-tenant operator that isolates resources across different payment processing environments.**

*Expected Answer Points:*
- Namespace-scoped vs cluster-scoped operators
- RBAC design for multi-tenancy
- Resource quota management
- Cross-tenant security isolation
- Tenant-specific configuration management

#### **Service Mesh & Networking (Staff Level)**

**Q4: Design a service mesh architecture for a payment processing system with strict security and compliance requirements.**

*Expected Answer:*
```
- Istio service mesh with strict mTLS enforcement
- Authorization policies for payment endpoints
- Traffic routing with canary deployments
- Circuit breaker patterns for external APIs
- End-to-end encryption for sensitive data
- Audit logging for compliance
- Multi-cluster service discovery
```

**Q5: How would you implement zero-downtime deployments for a critical payment service using Istio?**

*Expected Answer Points:*
- Virtual Service configuration for traffic splitting
- Destination Rules with health checks
- Progressive delivery with automated rollback
- Metrics-based success criteria
- Integration with monitoring and alerting

**Q6: Debug a scenario where payment transactions are failing intermittently in a service mesh environment.**

*Expected Approach:*
- Analyze Envoy proxy logs and metrics
- Check service mesh configuration (VS, DR, AuthZ policies)
- Validate certificate rotation and mTLS configuration
- Examine network policies and firewall rules
- Use distributed tracing to identify bottlenecks

#### **Security & RBAC (Principal Level)**

**Q7: Design a comprehensive security model for a Kubernetes-based payment processing platform.**

*Expected Architecture:*
```
1. Multi-layered RBAC with service accounts
2. Network policies for traffic isolation
3. Pod Security Policies/Pod Security Standards
4. Secret management with external secret stores
5. Admission controllers for policy enforcement
6. Runtime security monitoring
7. Compliance audit trails
```

**Q8: Implement automated secret rotation for banking partner credentials without service disruption.**

*Expected Solution:*
- External Secrets Operator integration
- Rolling updates with graceful shutdown
- Health checks during rotation
- Fallback mechanisms for failed rotations
- Audit logging for compliance

#### **Multi-Cluster & GitOps (Architect Level)**

**Q9: Design a disaster recovery strategy for a payment processing system spanning multiple Kubernetes clusters.**

*Expected Components:*
```
1. Cross-cluster replication strategy
2. Data consistency and synchronization
3. Automated failover mechanisms
4. Traffic routing and load balancing
5. Backup and restore procedures
6. RTO/RPO requirements compliance
7. Testing and validation procedures
```

**Q10: Implement a GitOps workflow for managing payment system deployments across dev, staging, and production environments.**

*Expected Implementation:*
- Git repository structure with environment branching
- ArgoCD/Flux application management
- Progressive deployment pipeline
- Environment promotion strategies
- Configuration management and secrets handling
- Rollback and emergency procedures

#### **Performance & Scaling (Distinguished Engineer Level)**

**Q11: Optimize Kubernetes resource allocation for a payment processing system handling 100,000 TPS.**

*Expected Optimization Areas:*
```
1. Resource requests/limits tuning
2. HPA/VPA configuration optimization
3. Node affinity and anti-affinity rules
4. Pod disruption budgets
5. Priority classes for critical workloads
6. Custom metrics for business-aware scaling
7. Resource quota management
8. Performance testing and benchmarking
```

**Q12: Design an auto-scaling strategy that considers both infrastructure metrics and business metrics (transaction volume, fraud score).**

*Expected Solution:*
- Custom metrics from application (transaction rate, queue depth)
- External metrics from business systems
- Predictive scaling based on historical patterns
- Multi-dimensional scaling policies
- Cost optimization considerations

### **Hands-On Scenarios**

#### **Scenario 1: Production Incident Response**

*Situation:* Payment processing is experiencing high latency (P95 > 5 seconds) and increasing error rates.

*Expected Actions:*
1. Check pod resource utilization and limits
2. Analyze service mesh metrics and circuit breaker status
3. Examine database connection pools and query performance
4. Review recent deployments and configuration changes
5. Implement emergency scaling if needed
6. Prepare rollback plan if necessary

#### **Scenario 2: Security Breach Investigation**

*Situation:* Suspicious API calls detected in payment processing namespace.

*Expected Investigation:*
1. Review admission controller logs
2. Analyze network policy violations
3. Check RBAC audit logs
4. Examine pod security context violations
5. Review service mesh authorization policies
6. Implement immediate containment measures

#### **Scenario 3: Capacity Planning**

*Situation:* Preparing for Black Friday traffic with expected 10x increase in payment volume.

*Expected Planning:*
1. Historical analysis of traffic patterns
2. Load testing with realistic scenarios
3. Resource capacity calculation and reservation
4. Auto-scaling configuration adjustment
5. Monitoring and alerting threshold updates
6. Disaster recovery plan validation

### **Best Practices Assessment**

**Q13: What are the key considerations when running stateful workloads in Kubernetes for payment processing?**

*Expected Points:*
- StatefulSet vs Deployment usage
- Persistent volume management and backup
- Data consistency and replication
- Ordered startup and shutdown
- Network identity and service discovery
- Disaster recovery and data migration

**Q14: How do you ensure compliance (PCI DSS, SOX) in a Kubernetes-based payment system?**

*Expected Compliance Measures:*
- Data encryption at rest and in transit
- Access logging and audit trails
- Segregation of duties through RBAC
- Regular security scanning and updates
- Incident response procedures
- Compliance monitoring and reporting

This comprehensive guide demonstrates enterprise-level Kubernetes operations expertise essential for senior infrastructure engineering roles at companies like Razorpay, covering advanced CRDs, operators, service mesh, security, multi-cluster management, and GitOps practices.

##  Gitops  Continuous Deployment

<!-- AUTO-GENERATED ANCHOR: originally referenced as #-gitops--continuous-deployment -->

Placeholder content. Please replace with proper section.


##  Disaster Recovery  Backup

<!-- AUTO-GENERATED ANCHOR: originally referenced as #-disaster-recovery--backup -->

Placeholder content. Please replace with proper section.


##  Interview Questions  Scenarios

<!-- AUTO-GENERATED ANCHOR: originally referenced as #-interview-questions--scenarios -->

Placeholder content. Please replace with proper section.
