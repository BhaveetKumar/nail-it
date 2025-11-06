---
# Auto-generated front matter
Title: Ml Backend Systems Guide
LastUpdated: 2025-11-06T20:45:58.310531
Tags: []
Status: draft
---

# 🤖 ML Backend Systems Guide

> **Complete guide to building machine learning backend systems and AI-powered applications**

## 📚 Table of Contents

1. [ML Backend Architecture](#-ml-backend-architecture)
2. [Model Serving](#-model-serving)
3. [Feature Engineering](#-feature-engineering)
4. [ML Pipelines](#-ml-pipelines)
5. [Real-time ML](#-real-time-ml)
6. [MLOps](#-mlops)
7. [Case Studies](#-case-studies)

---

## 🏗️ ML Backend Architecture

### ML System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ML Backend Architecture                  │
├─────────────────────────────────────────────────────────────┤
│  Client Apps    Web Apps    Mobile Apps    IoT Devices     │
│       │             │            │              │          │
│       └─────────────┼────────────┼──────────────┘          │
│                     │            │                         │
│            ┌────────▼────────────▼────────┐                │
│            │      API Gateway             │                │
│            └─────────┬────────────────────┘                │
│                      │                                     │
│    ┌─────────────────┼─────────────────┐                   │
│    │                 │                 │                   │
│ ┌──▼──┐          ┌───▼───┐         ┌───▼───┐               │
│ │ ML  │          │ Data  │         │ Model │               │
│ │ API │          │ API   │         │ API   │               │
│ └─────┘          └───────┘         └───────┘               │
│    │                 │                 │                   │
│    └─────────────────┼─────────────────┘                   │
│                      │                                     │
│            ┌─────────▼───────────┐                         │
│            │   ML Pipeline       │                         │
│            │   - Data Ingestion  │                         │
│            │   - Feature Store   │                         │
│            │   - Model Training  │                         │
│            │   - Model Serving   │                         │
│            └─────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### ML Backend Patterns

1. **Batch Processing**: Process data in batches
2. **Stream Processing**: Real-time data processing
3. **Model Serving**: Serve ML models via APIs
4. **Feature Store**: Centralized feature management
5. **A/B Testing**: Experiment with different models

---

## 🚀 Model Serving

### 1. REST API Model Serving

```go
// ML Model Server
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "sync"
    "time"
)

type ModelServer struct {
    models map[string]MLModel
    mutex  sync.RWMutex
}

type MLModel interface {
    Predict(input interface{}) (interface{}, error)
    GetMetadata() ModelMetadata
}

type ModelMetadata struct {
    Name        string    `json:"name"`
    Version     string    `json:"version"`
    CreatedAt   time.Time `json:"created_at"`
    InputSchema interface{} `json:"input_schema"`
    OutputSchema interface{} `json:"output_schema"`
}

type PredictionRequest struct {
    ModelName string      `json:"model_name"`
    Input     interface{} `json:"input"`
}

type PredictionResponse struct {
    Prediction interface{} `json:"prediction"`
    Confidence float64     `json:"confidence"`
    ModelInfo  ModelMetadata `json:"model_info"`
}

func NewModelServer() *ModelServer {
    return &ModelServer{
        models: make(map[string]MLModel),
    }
}

func (ms *ModelServer) RegisterModel(name string, model MLModel) {
    ms.mutex.Lock()
    defer ms.mutex.Unlock()
    ms.models[name] = model
}

func (ms *ModelServer) Predict(w http.ResponseWriter, r *http.Request) {
    var req PredictionRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid request body", http.StatusBadRequest)
        return
    }

    ms.mutex.RLock()
    model, exists := ms.models[req.ModelName]
    ms.mutex.RUnlock()

    if !exists {
        http.Error(w, "Model not found", http.StatusNotFound)
        return
    }

    prediction, err := model.Predict(req.Input)
    if err != nil {
        http.Error(w, "Prediction failed", http.StatusInternalServerError)
        return
    }

    response := PredictionResponse{
        Prediction: prediction,
        Confidence: 0.95, // Example confidence score
        ModelInfo:  model.GetMetadata(),
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func (ms *ModelServer) ListModels(w http.ResponseWriter, r *http.Request) {
    ms.mutex.RLock()
    defer ms.mutex.RUnlock()

    models := make([]ModelMetadata, 0, len(ms.models))
    for _, model := range ms.models {
        models = append(models, model.GetMetadata())
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(models)
}

// Example ML Model Implementation
type FraudDetectionModel struct {
    metadata ModelMetadata
}

func NewFraudDetectionModel() *FraudDetectionModel {
    return &FraudDetectionModel{
        metadata: ModelMetadata{
            Name:      "fraud_detection",
            Version:   "1.0.0",
            CreatedAt: time.Now(),
        },
    }
}

func (fdm *FraudDetectionModel) Predict(input interface{}) (interface{}, error) {
    // Convert input to features
    features, ok := input.(map[string]interface{})
    if !ok {
        return nil, fmt.Errorf("invalid input format")
    }

    // Extract features
    amount, _ := features["amount"].(float64)
    userAge, _ := features["user_age"].(float64)
    transactionCount, _ := features["transaction_count"].(float64)

    // Simple fraud detection logic
    fraudScore := 0.0
    if amount > 10000 {
        fraudScore += 0.3
    }
    if userAge < 25 {
        fraudScore += 0.2
    }
    if transactionCount > 100 {
        fraudScore += 0.1
    }

    return map[string]interface{}{
        "fraud_probability": fraudScore,
        "is_fraud":          fraudScore > 0.5,
    }, nil
}

func (fdm *FraudDetectionModel) GetMetadata() ModelMetadata {
    return fdm.metadata
}

func main() {
    server := NewModelServer()
    
    // Register models
    server.RegisterModel("fraud_detection", NewFraudDetectionModel())
    
    // Setup routes
    http.HandleFunc("/predict", server.Predict)
    http.HandleFunc("/models", server.ListModels)
    
    fmt.Println("ML Model Server starting on :8080")
    http.ListenAndServe(":8080", nil)
}
```

### 2. gRPC Model Serving

```go
// gRPC ML Model Server
package main

import (
    "context"
    "fmt"
    "log"
    "net"

    "google.golang.org/grpc"
    "google.golang.org/grpc/reflection"
)

// Define the service
type MLServiceServer struct {
    models map[string]MLModel
}

func (s *MLServiceServer) Predict(ctx context.Context, req *PredictionRequest) (*PredictionResponse, error) {
    model, exists := s.models[req.ModelName]
    if !exists {
        return nil, fmt.Errorf("model not found: %s", req.ModelName)
    }

    prediction, err := model.Predict(req.Input)
    if err != nil {
        return nil, fmt.Errorf("prediction failed: %v", err)
    }

    return &PredictionResponse{
        Prediction: prediction,
        Confidence: 0.95,
    }, nil
}

func (s *MLServiceServer) GetModelInfo(ctx context.Context, req *ModelInfoRequest) (*ModelInfoResponse, error) {
    model, exists := s.models[req.ModelName]
    if !exists {
        return nil, fmt.Errorf("model not found: %s", req.ModelName)
    }

    metadata := model.GetMetadata()
    return &ModelInfoResponse{
        Name:        metadata.Name,
        Version:     metadata.Version,
        CreatedAt:   metadata.CreatedAt.Unix(),
    }, nil
}

func main() {
    lis, err := net.Listen("tcp", ":50051")
    if err != nil {
        log.Fatalf("failed to listen: %v", err)
    }

    s := grpc.NewServer()
    mlServer := &MLServiceServer{
        models: make(map[string]MLModel),
    }
    
    // Register models
    mlServer.models["fraud_detection"] = NewFraudDetectionModel()
    
    RegisterMLServiceServer(s, mlServer)
    reflection.Register(s)

    log.Println("gRPC ML Server starting on :50051")
    if err := s.Serve(lis); err != nil {
        log.Fatalf("failed to serve: %v", err)
    }
}
```

---

## 🔧 Feature Engineering

### 1. Feature Store

```go
// Feature Store Implementation
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "sync"
    "time"
)

type FeatureStore struct {
    features map[string]map[string]Feature
    mutex    sync.RWMutex
}

type Feature struct {
    Name        string      `json:"name"`
    Value       interface{} `json:"value"`
    Timestamp   time.Time   `json:"timestamp"`
    TTL         time.Duration `json:"ttl"`
}

type FeatureRequest struct {
    EntityID    string   `json:"entity_id"`
    FeatureNames []string `json:"feature_names"`
}

type FeatureResponse struct {
    EntityID string              `json:"entity_id"`
    Features map[string]Feature  `json:"features"`
}

func NewFeatureStore() *FeatureStore {
    fs := &FeatureStore{
        features: make(map[string]map[string]Feature),
    }
    
    // Start cleanup goroutine
    go fs.cleanup()
    
    return fs
}

func (fs *FeatureStore) SetFeature(entityID, featureName string, value interface{}, ttl time.Duration) {
    fs.mutex.Lock()
    defer fs.mutex.Unlock()
    
    if fs.features[entityID] == nil {
        fs.features[entityID] = make(map[string]Feature)
    }
    
    fs.features[entityID][featureName] = Feature{
        Name:      featureName,
        Value:     value,
        Timestamp: time.Now(),
        TTL:       ttl,
    }
}

func (fs *FeatureStore) GetFeatures(ctx context.Context, req FeatureRequest) (*FeatureResponse, error) {
    fs.mutex.RLock()
    defer fs.mutex.RUnlock()
    
    entityFeatures, exists := fs.features[req.EntityID]
    if !exists {
        return &FeatureResponse{
            EntityID: req.EntityID,
            Features: make(map[string]Feature),
        }, nil
    }
    
    features := make(map[string]Feature)
    for _, featureName := range req.FeatureNames {
        if feature, exists := entityFeatures[featureName]; exists {
            // Check if feature is still valid
            if time.Since(feature.Timestamp) < feature.TTL {
                features[featureName] = feature
            }
        }
    }
    
    return &FeatureResponse{
        EntityID: req.EntityID,
        Features: features,
    }, nil
}

func (fs *FeatureStore) cleanup() {
    ticker := time.NewTicker(1 * time.Minute)
    defer ticker.Stop()
    
    for range ticker.C {
        fs.mutex.Lock()
        now := time.Now()
        
        for entityID, entityFeatures := range fs.features {
            for featureName, feature := range entityFeatures {
                if now.Sub(feature.Timestamp) > feature.TTL {
                    delete(entityFeatures, featureName)
                }
            }
            
            if len(entityFeatures) == 0 {
                delete(fs.features, entityID)
            }
        }
        
        fs.mutex.Unlock()
    }
}

// Feature Engineering Pipeline
type FeaturePipeline struct {
    steps []FeatureStep
}

type FeatureStep interface {
    Process(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error)
}

type FeatureStepFunc func(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error)

func (f FeatureStepFunc) Process(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
    return f(ctx, input)
}

func NewFeaturePipeline() *FeaturePipeline {
    return &FeaturePipeline{
        steps: make([]FeatureStep, 0),
    }
}

func (fp *FeaturePipeline) AddStep(step FeatureStep) *FeaturePipeline {
    fp.steps = append(fp.steps, step)
    return fp
}

func (fp *FeaturePipeline) Process(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
    result := input
    
    for _, step := range fp.steps {
        var err error
        result, err = step.Process(ctx, result)
        if err != nil {
            return nil, fmt.Errorf("feature step failed: %v", err)
        }
    }
    
    return result, nil
}

// Example feature steps
func NormalizeFeature(featureName string) FeatureStep {
    return FeatureStepFunc(func(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
        if value, exists := input[featureName]; exists {
            if num, ok := value.(float64); ok {
                // Simple normalization (0-1)
                input[featureName+"_normalized"] = num / 100.0
            }
        }
        return input, nil
    })
}

func CreateInteractionFeature(feature1, feature2, outputName string) FeatureStep {
    return FeatureStepFunc(func(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
        val1, ok1 := input[feature1].(float64)
        val2, ok2 := input[feature2].(float64)
        
        if ok1 && ok2 {
            input[outputName] = val1 * val2
        }
        
        return input, nil
    })
}
```

---

## 🔄 ML Pipelines

### 1. Training Pipeline

```go
// ML Training Pipeline
package main

import (
    "context"
    "fmt"
    "time"
)

type TrainingPipeline struct {
    dataSource    DataSource
    preprocessor  Preprocessor
    trainer       Trainer
    evaluator     Evaluator
    modelStore    ModelStore
}

type DataSource interface {
    LoadData(ctx context.Context) (Dataset, error)
}

type Preprocessor interface {
    Process(ctx context.Context, data Dataset) (Dataset, error)
}

type Trainer interface {
    Train(ctx context.Context, data Dataset) (Model, error)
}

type Evaluator interface {
    Evaluate(ctx context.Context, model Model, data Dataset) (EvaluationResult, error)
}

type ModelStore interface {
    SaveModel(ctx context.Context, model Model) error
}

type Dataset struct {
    Features [][]float64
    Labels   []float64
    Metadata map[string]interface{}
}

type Model interface {
    Predict(features []float64) (float64, error)
    GetMetadata() ModelMetadata
}

type EvaluationResult struct {
    Accuracy  float64
    Precision float64
    Recall    float64
    F1Score   float64
}

func NewTrainingPipeline(
    dataSource DataSource,
    preprocessor Preprocessor,
    trainer Trainer,
    evaluator Evaluator,
    modelStore ModelStore,
) *TrainingPipeline {
    return &TrainingPipeline{
        dataSource:   dataSource,
        preprocessor: preprocessor,
        trainer:      trainer,
        evaluator:    evaluator,
        modelStore:   modelStore,
    }
}

func (tp *TrainingPipeline) Run(ctx context.Context) error {
    // 1. Load data
    fmt.Println("Loading data...")
    data, err := tp.dataSource.LoadData(ctx)
    if err != nil {
        return fmt.Errorf("failed to load data: %v", err)
    }

    // 2. Preprocess data
    fmt.Println("Preprocessing data...")
    processedData, err := tp.preprocessor.Process(ctx, data)
    if err != nil {
        return fmt.Errorf("failed to preprocess data: %v", err)
    }

    // 3. Split data
    trainData, testData := tp.splitData(processedData, 0.8)

    // 4. Train model
    fmt.Println("Training model...")
    model, err := tp.trainer.Train(ctx, trainData)
    if err != nil {
        return fmt.Errorf("failed to train model: %v", err)
    }

    // 5. Evaluate model
    fmt.Println("Evaluating model...")
    evalResult, err := tp.evaluator.Evaluate(ctx, model, testData)
    if err != nil
        return fmt.Errorf("failed to evaluate model: %v", err)
    }

    fmt.Printf("Model evaluation: Accuracy=%.3f, Precision=%.3f, Recall=%.3f, F1=%.3f\n",
        evalResult.Accuracy, evalResult.Precision, evalResult.Recall, evalResult.F1Score)

    // 6. Save model
    fmt.Println("Saving model...")
    if err := tp.modelStore.SaveModel(ctx, model); err != nil {
        return fmt.Errorf("failed to save model: %v", err)
    }

    return nil
}

func (tp *TrainingPipeline) splitData(data Dataset, ratio float64) (Dataset, Dataset) {
    splitIndex := int(float64(len(data.Features)) * ratio)
    
    trainData := Dataset{
        Features: data.Features[:splitIndex],
        Labels:   data.Labels[:splitIndex],
        Metadata: data.Metadata,
    }
    
    testData := Dataset{
        Features: data.Features[splitIndex:],
        Labels:   data.Labels[splitIndex:],
        Metadata: data.Metadata,
    }
    
    return trainData, testData
}
```

### 2. Inference Pipeline

```go
// ML Inference Pipeline
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type InferencePipeline struct {
    featureStore FeatureStore
    modelStore   ModelStore
    preprocessor Preprocessor
    models       map[string]Model
    mutex        sync.RWMutex
}

func NewInferencePipeline(
    featureStore FeatureStore,
    modelStore ModelStore,
    preprocessor Preprocessor,
) *InferencePipeline {
    return &InferencePipeline{
        featureStore: featureStore,
        modelStore:   modelStore,
        preprocessor: preprocessor,
        models:       make(map[string]Model),
    }
}

func (ip *InferencePipeline) Predict(ctx context.Context, req PredictionRequest) (*PredictionResponse, error) {
    // 1. Get features
    features, err := ip.getFeatures(ctx, req.EntityID, req.FeatureNames)
    if err != nil {
        return nil, fmt.Errorf("failed to get features: %v", err)
    }

    // 2. Preprocess features
    processedFeatures, err := ip.preprocessor.Process(ctx, features)
    if err != nil {
        return nil, fmt.Errorf("failed to preprocess features: %v", err)
    }

    // 3. Get model
    model, err := ip.getModel(ctx, req.ModelName)
    if err != nil {
        return nil, fmt.Errorf("failed to get model: %v", err)
    }

    // 4. Make prediction
    prediction, err := model.Predict(processedFeatures)
    if err != nil {
        return nil, fmt.Errorf("failed to make prediction: %v", err)
    }

    return &PredictionResponse{
        Prediction: prediction,
        Confidence: 0.95, // Example confidence
        ModelInfo:  model.GetMetadata(),
    }, nil
}

func (ip *InferencePipeline) getFeatures(ctx context.Context, entityID string, featureNames []string) (map[string]interface{}, error) {
    req := FeatureRequest{
        EntityID:     entityID,
        FeatureNames: featureNames,
    }
    
    resp, err := ip.featureStore.GetFeatures(ctx, req)
    if err != nil {
        return nil, err
    }
    
    features := make(map[string]interface{})
    for name, feature := range resp.Features {
        features[name] = feature.Value
    }
    
    return features, nil
}

func (ip *InferencePipeline) getModel(ctx context.Context, modelName string) (Model, error) {
    ip.mutex.RLock()
    model, exists := ip.models[modelName]
    ip.mutex.RUnlock()
    
    if exists {
        return model, nil
    }
    
    // Load model from store
    model, err := ip.modelStore.LoadModel(ctx, modelName)
    if err != nil {
        return nil, err
    }
    
    ip.mutex.Lock()
    ip.models[modelName] = model
    ip.mutex.Unlock()
    
    return model, nil
}
```

---

## ⚡ Real-time ML

### 1. Stream Processing

```go
// Real-time ML with Kafka Streams
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "time"
)

type StreamProcessor struct {
    featureStore FeatureStore
    modelStore   ModelStore
    producer     EventProducer
    consumer     EventConsumer
}

type EventProducer interface {
    Publish(ctx context.Context, topic string, event interface{}) error
}

type EventConsumer interface {
    Subscribe(ctx context.Context, topic string, handler EventHandler) error
}

type EventHandler interface {
    Handle(ctx context.Context, event interface{}) error
}

type MLStreamProcessor struct {
    featureStore FeatureStore
    modelStore   ModelStore
    producer     EventProducer
}

func NewMLStreamProcessor(
    featureStore FeatureStore,
    modelStore ModelStore,
    producer EventProducer,
) *MLStreamProcessor {
    return &MLStreamProcessor{
        featureStore: featureStore,
        modelStore:   modelStore,
        producer:     producer,
    }
}

func (msp *MLStreamProcessor) ProcessTransaction(ctx context.Context, event TransactionEvent) error {
    // 1. Extract features
    features := msp.extractFeatures(event)
    
    // 2. Store features
    for name, value := range features {
        msp.featureStore.SetFeature(
            event.UserID,
            name,
            value,
            24*time.Hour, // TTL
        )
    }
    
    // 3. Get model
    model, err := msp.modelStore.LoadModel(ctx, "fraud_detection")
    if err != nil {
        return fmt.Errorf("failed to load model: %v", err)
    }
    
    // 4. Make prediction
    prediction, err := model.Predict(features)
    if err != nil {
        return fmt.Errorf("failed to make prediction: %v", err)
    }
    
    // 5. Publish result
    result := PredictionResult{
        TransactionID: event.TransactionID,
        UserID:        event.UserID,
        Prediction:    prediction,
        Timestamp:     time.Now(),
    }
    
    return msp.producer.Publish(ctx, "predictions", result)
}

func (msp *MLStreamProcessor) extractFeatures(event TransactionEvent) map[string]interface{} {
    return map[string]interface{}{
        "amount":              event.Amount,
        "user_age":            event.UserAge,
        "transaction_count":   event.TransactionCount,
        "merchant_category":   event.MerchantCategory,
        "time_of_day":         event.Timestamp.Hour(),
        "day_of_week":         event.Timestamp.Weekday(),
    }
}

type TransactionEvent struct {
    TransactionID     string    `json:"transaction_id"`
    UserID           string    `json:"user_id"`
    Amount           float64   `json:"amount"`
    UserAge          int       `json:"user_age"`
    TransactionCount int       `json:"transaction_count"`
    MerchantCategory string    `json:"merchant_category"`
    Timestamp        time.Time `json:"timestamp"`
}

type PredictionResult struct {
    TransactionID string      `json:"transaction_id"`
    UserID        string      `json:"user_id"`
    Prediction    interface{} `json:"prediction"`
    Timestamp     time.Time   `json:"timestamp"`
}
```

---

## 🔧 MLOps

### 1. Model Versioning

```go
// Model Versioning System
package main

import (
    "context"
    "fmt"
    "time"
)

type ModelVersion struct {
    ID          string
    ModelName   string
    Version     string
    Path        string
    CreatedAt   time.Time
    Metadata    map[string]interface{}
    Performance EvaluationResult
}

type ModelRegistry struct {
    versions map[string][]ModelVersion
    mutex    sync.RWMutex
}

func NewModelRegistry() *ModelRegistry {
    return &ModelRegistry{
        versions: make(map[string][]ModelVersion),
    }
}

func (mr *ModelRegistry) RegisterModel(ctx context.Context, version ModelVersion) error {
    mr.mutex.Lock()
    defer mr.mutex.Unlock()
    
    mr.versions[version.ModelName] = append(mr.versions[version.ModelName], version)
    return nil
}

func (mr *ModelRegistry) GetLatestVersion(ctx context.Context, modelName string) (*ModelVersion, error) {
    mr.mutex.RLock()
    defer mr.mutex.RUnlock()
    
    versions, exists := mr.versions[modelName]
    if !exists || len(versions) == 0 {
        return nil, fmt.Errorf("no versions found for model: %s", modelName)
    }
    
    latest := versions[0]
    for _, version := range versions[1:] {
        if version.CreatedAt.After(latest.CreatedAt) {
            latest = version
        }
    }
    
    return &latest, nil
}

func (mr *ModelRegistry) GetVersion(ctx context.Context, modelName, version string) (*ModelVersion, error) {
    mr.mutex.RLock()
    defer mr.mutex.RUnlock()
    
    versions, exists := mr.versions[modelName]
    if !exists {
        return nil, fmt.Errorf("model not found: %s", modelName)
    }
    
    for _, v := range versions {
        if v.Version == version {
            return &v, nil
        }
    }
    
    return nil, fmt.Errorf("version not found: %s@%s", modelName, version)
}
```

### 2. Model Monitoring

```go
// Model Monitoring System
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type ModelMonitor struct {
    metrics map[string]ModelMetrics
    mutex   sync.RWMutex
}

type ModelMetrics struct {
    ModelName     string
    Version       string
    RequestCount  int64
    ErrorCount    int64
    AvgLatency    time.Duration
    LastRequest   time.Time
    Performance   EvaluationResult
}

func NewModelMonitor() *ModelMonitor {
    return &ModelMonitor{
        metrics: make(map[string]ModelMetrics),
    }
}

func (mm *ModelMonitor) RecordRequest(ctx context.Context, modelName, version string, latency time.Duration, success bool) {
    mm.mutex.Lock()
    defer mm.mutex.Unlock()
    
    key := fmt.Sprintf("%s@%s", modelName, version)
    metrics := mm.metrics[key]
    
    metrics.ModelName = modelName
    metrics.Version = version
    metrics.RequestCount++
    metrics.LastRequest = time.Now()
    
    if !success {
        metrics.ErrorCount++
    }
    
    // Update average latency
    if metrics.AvgLatency == 0 {
        metrics.AvgLatency = latency
    } else {
        metrics.AvgLatency = (metrics.AvgLatency + latency) / 2
    }
    
    mm.metrics[key] = metrics
}

func (mm *ModelMonitor) GetMetrics(ctx context.Context, modelName, version string) (*ModelMetrics, error) {
    mm.mutex.RLock()
    defer mm.mutex.RUnlock()
    
    key := fmt.Sprintf("%s@%s", modelName, version)
    metrics, exists := mm.metrics[key]
    if !exists {
        return nil, fmt.Errorf("metrics not found for model: %s@%s", modelName, version)
    }
    
    return &metrics, nil
}

func (mm *ModelMonitor) CheckHealth(ctx context.Context, modelName, version string) (*HealthStatus, error) {
    metrics, err := mm.GetMetrics(ctx, modelName, version)
    if err != nil {
        return nil, err
    }
    
    status := &HealthStatus{
        ModelName: modelName,
        Version:   version,
        Status:    "healthy",
        Issues:    make([]string, 0),
    }
    
    // Check error rate
    if metrics.RequestCount > 0 {
        errorRate := float64(metrics.ErrorCount) / float64(metrics.RequestCount)
        if errorRate > 0.1 { // 10% error rate threshold
            status.Status = "unhealthy"
            status.Issues = append(status.Issues, fmt.Sprintf("High error rate: %.2f%%", errorRate*100))
        }
    }
    
    // Check latency
    if metrics.AvgLatency > 1*time.Second {
        status.Status = "unhealthy"
        status.Issues = append(status.Issues, fmt.Sprintf("High latency: %v", metrics.AvgLatency))
    }
    
    // Check staleness
    if time.Since(metrics.LastRequest) > 1*time.Hour {
        status.Status = "stale"
        status.Issues = append(status.Issues, "No requests in the last hour")
    }
    
    return status, nil
}

type HealthStatus struct {
    ModelName string   `json:"model_name"`
    Version   string   `json:"version"`
    Status    string   `json:"status"`
    Issues    []string `json:"issues"`
}
```

---

## 🎯 Case Studies

### Case Study 1: Fraud Detection System

**Problem**: Detect fraudulent transactions in real-time

**Solution**:
1. **Feature Engineering**: Extract transaction features
2. **Model Training**: Train fraud detection model
3. **Real-time Inference**: Serve predictions via API
4. **Monitoring**: Track model performance

**Results**:
- 95% fraud detection accuracy
- <100ms prediction latency
- 99.9% system uptime

### Case Study 2: Recommendation System

**Problem**: Recommend products to users

**Solution**:
1. **Collaborative Filtering**: User-item interactions
2. **Content-based Filtering**: Product features
3. **Hybrid Approach**: Combine both methods
4. **A/B Testing**: Experiment with different models

**Results**:
- 30% increase in click-through rate
- 25% increase in conversion rate
- 40% improvement in user engagement

---

## 🎯 Best Practices

### 1. Model Serving
- **Versioning**: Implement model versioning
- **Rollback**: Support model rollback
- **Monitoring**: Monitor model performance
- **Scaling**: Auto-scale based on load

### 2. Feature Engineering
- **Feature Store**: Centralized feature management
- **Feature Validation**: Validate feature quality
- **Feature Monitoring**: Monitor feature drift
- **Feature Reuse**: Reuse features across models

### 3. MLOps
- **CI/CD**: Automated model deployment
- **Testing**: Comprehensive model testing
- **Monitoring**: Real-time model monitoring
- **Governance**: Model governance and compliance

---

**🤖 Master these ML backend patterns to build intelligent, scalable AI systems! 🚀**
