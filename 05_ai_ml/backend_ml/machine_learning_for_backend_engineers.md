# 🤖 Machine Learning for Backend Engineers

## Table of Contents
1. [ML Fundamentals for Backend](#ml-fundamentals-for-backend)
2. [Model Serving & APIs](#model-serving--apis)
3. [Feature Engineering](#feature-engineering)
4. [Model Deployment](#model-deployment)
5. [MLOps for Backend](#mlops-for-backend)
6. [Real-time ML Systems](#real-time-ml-systems)
7. [ML Infrastructure](#ml-infrastructure)
8. [Go Implementation Examples](#go-implementation-examples)
9. [Interview Questions](#interview-questions)

## ML Fundamentals for Backend

### Understanding ML in Backend Context

```go
package main

import (
    "encoding/json"
    "fmt"
    "math"
    "time"
)

// ML Model Interface
type MLModel interface {
    Predict(input []float64) ([]float64, error)
    Train(data [][]float64, labels []float64) error
    Save(path string) error
    Load(path string) error
}

// Linear Regression Model
type LinearRegression struct {
    Weights []float64
    Bias    float64
    LearningRate float64
    Epochs  int
}

func NewLinearRegression(learningRate float64, epochs int) *LinearRegression {
    return &LinearRegression{
        LearningRate: learningRate,
        Epochs:       epochs,
    }
}

func (lr *LinearRegression) Predict(input []float64) ([]float64, error) {
    if len(input) != len(lr.Weights) {
        return nil, fmt.Errorf("input dimension mismatch")
    }
    
    prediction := lr.Bias
    for i, weight := range lr.Weights {
        prediction += weight * input[i]
    }
    
    return []float64{prediction}, nil
}

func (lr *LinearRegression) Train(data [][]float64, labels []float64) error {
    if len(data) == 0 || len(data[0]) == 0 {
        return fmt.Errorf("empty training data")
    }
    
    features := len(data[0])
    lr.Weights = make([]float64, features)
    
    for epoch := 0; epoch < lr.Epochs; epoch++ {
        for i, sample := range data {
            prediction, _ := lr.Predict(sample)
            error := labels[i] - prediction[0]
            
            // Update bias
            lr.Bias += lr.LearningRate * error
            
            // Update weights
            for j := range lr.Weights {
                lr.Weights[j] += lr.LearningRate * error * sample[j]
            }
        }
    }
    
    return nil
}

func (lr *LinearRegression) Save(path string) error {
    modelData := map[string]interface{}{
        "weights": lr.Weights,
        "bias":    lr.Bias,
        "type":    "linear_regression",
    }
    
    data, err := json.Marshal(modelData)
    if err != nil {
        return err
    }
    
    return os.WriteFile(path, data, 0644)
}

func (lr *LinearRegression) Load(path string) error {
    data, err := os.ReadFile(path)
    if err != nil {
        return err
    }
    
    var modelData map[string]interface{}
    if err := json.Unmarshal(data, &modelData); err != nil {
        return err
    }
    
    // Load weights
    if weights, ok := modelData["weights"].([]interface{}); ok {
        lr.Weights = make([]float64, len(weights))
        for i, w := range weights {
            if weight, ok := w.(float64); ok {
                lr.Weights[i] = weight
            }
        }
    }
    
    // Load bias
    if bias, ok := modelData["bias"].(float64); ok {
        lr.Bias = bias
    }
    
    return nil
}

// Logistic Regression for Classification
type LogisticRegression struct {
    Weights []float64
    Bias    float64
    LearningRate float64
    Epochs  int
}

func NewLogisticRegression(learningRate float64, epochs int) *LogisticRegression {
    return &LogisticRegression{
        LearningRate: learningRate,
        Epochs:       epochs,
    }
}

func (lr *LogisticRegression) Sigmoid(x float64) float64 {
    return 1.0 / (1.0 + math.Exp(-x))
}

func (lr *LogisticRegression) Predict(input []float64) ([]float64, error) {
    if len(input) != len(lr.Weights) {
        return nil, fmt.Errorf("input dimension mismatch")
    }
    
    z := lr.Bias
    for i, weight := range lr.Weights {
        z += weight * input[i]
    }
    
    probability := lr.Sigmoid(z)
    return []float64{probability}, nil
}

func (lr *LogisticRegression) Train(data [][]float64, labels []float64) error {
    if len(data) == 0 || len(data[0]) == 0 {
        return fmt.Errorf("empty training data")
    }
    
    features := len(data[0])
    lr.Weights = make([]float64, features)
    
    for epoch := 0; epoch < lr.Epochs; epoch++ {
        for i, sample := range data {
            prediction, _ := lr.Predict(sample)
            error := labels[i] - prediction[0]
            
            // Update bias
            lr.Bias += lr.LearningRate * error
            
            // Update weights
            for j := range lr.Weights {
                lr.Weights[j] += lr.LearningRate * error * sample[j]
            }
        }
    }
    
    return nil
}
```

## Model Serving & APIs

### ML Model Server

```go
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
    cache  map[string]CacheEntry
}

type CacheEntry struct {
    Prediction []float64
    Timestamp  time.Time
    TTL        time.Duration
}

type PredictionRequest struct {
    ModelName string    `json:"model_name"`
    Features  []float64 `json:"features"`
    UseCache  bool      `json:"use_cache,omitempty"`
}

type PredictionResponse struct {
    Prediction []float64 `json:"prediction"`
    ModelName  string    `json:"model_name"`
    Cached     bool      `json:"cached"`
    Latency    int64     `json:"latency_ms"`
}

func NewModelServer() *ModelServer {
    server := &ModelServer{
        models: make(map[string]MLModel),
        cache:  make(map[string]CacheEntry),
    }
    
    // Start cache cleanup goroutine
    go server.cleanupCache()
    
    return server
}

func (ms *ModelServer) RegisterModel(name string, model MLModel) {
    ms.mutex.Lock()
    defer ms.mutex.Unlock()
    ms.models[name] = model
}

func (ms *ModelServer) Predict(req PredictionRequest) (*PredictionResponse, error) {
    start := time.Now()
    
    // Check cache if enabled
    if req.UseCache {
        if cached, exists := ms.getFromCache(req.ModelName, req.Features); exists {
            return &PredictionResponse{
                Prediction: cached.Prediction,
                ModelName:  req.ModelName,
                Cached:     true,
                Latency:    time.Since(start).Milliseconds(),
            }, nil
        }
    }
    
    // Get model
    ms.mutex.RLock()
    model, exists := ms.models[req.ModelName]
    ms.mutex.RUnlock()
    
    if !exists {
        return nil, fmt.Errorf("model not found: %s", req.ModelName)
    }
    
    // Make prediction
    prediction, err := model.Predict(req.Features)
    if err != nil {
        return nil, err
    }
    
    // Cache result if enabled
    if req.UseCache {
        ms.setCache(req.ModelName, req.Features, prediction)
    }
    
    return &PredictionResponse{
        Prediction: prediction,
        ModelName:  req.ModelName,
        Cached:     false,
        Latency:    time.Since(start).Milliseconds(),
    }, nil
}

func (ms *ModelServer) getFromCache(modelName string, features []float64) (CacheEntry, bool) {
    key := ms.generateCacheKey(modelName, features)
    ms.mutex.RLock()
    defer ms.mutex.RUnlock()
    
    entry, exists := ms.cache[key]
    if !exists {
        return CacheEntry{}, false
    }
    
    // Check if expired
    if time.Since(entry.Timestamp) > entry.TTL {
        return CacheEntry{}, false
    }
    
    return entry, true
}

func (ms *ModelServer) setCache(modelName string, features []float64, prediction []float64) {
    key := ms.generateCacheKey(modelName, features)
    ms.mutex.Lock()
    defer ms.mutex.Unlock()
    
    ms.cache[key] = CacheEntry{
        Prediction: prediction,
        Timestamp:  time.Now(),
        TTL:        5 * time.Minute, // 5 minute TTL
    }
}

func (ms *ModelServer) generateCacheKey(modelName string, features []float64) string {
    // Simple hash-based cache key
    hash := 0
    for _, f := range features {
        hash = hash*31 + int(f*1000) // Scale to avoid floating point issues
    }
    return fmt.Sprintf("%s_%d", modelName, hash)
}

func (ms *ModelServer) cleanupCache() {
    ticker := time.NewTicker(1 * time.Minute)
    defer ticker.Stop()
    
    for range ticker.C {
        ms.mutex.Lock()
        now := time.Now()
        for key, entry := range ms.cache {
            if now.Sub(entry.Timestamp) > entry.TTL {
                delete(ms.cache, key)
            }
        }
        ms.mutex.Unlock()
    }
}

// HTTP Handler
func (ms *ModelServer) HandlePrediction(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }
    
    var req PredictionRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }
    
    resp, err := ms.Predict(req)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(resp)
}

// Batch Prediction
func (ms *ModelServer) BatchPredict(modelName string, inputs [][]float64) ([][]float64, error) {
    ms.mutex.RLock()
    model, exists := ms.models[modelName]
    ms.mutex.RUnlock()
    
    if !exists {
        return nil, fmt.Errorf("model not found: %s", modelName)
    }
    
    predictions := make([][]float64, len(inputs))
    for i, input := range inputs {
        pred, err := model.Predict(input)
        if err != nil {
            return nil, fmt.Errorf("prediction failed for input %d: %v", i, err)
        }
        predictions[i] = pred
    }
    
    return predictions, nil
}
```

## Feature Engineering

### Feature Engineering Pipeline

```go
package main

import (
    "math"
    "sort"
    "strings"
)

type FeatureEngineer struct {
    scalers map[string]*StandardScaler
    encoders map[string]*OneHotEncoder
}

type StandardScaler struct {
    Mean float64
    Std  float64
    Fitted bool
}

type OneHotEncoder struct {
    Categories map[string]int
    Fitted     bool
}

func NewFeatureEngineer() *FeatureEngineer {
    return &FeatureEngineer{
        scalers:  make(map[string]*StandardScaler),
        encoders: make(map[string]*OneHotEncoder),
    }
}

// Numerical Feature Scaling
func (fe *FeatureEngineer) FitStandardScaler(featureName string, data []float64) {
    if len(data) == 0 {
        return
    }
    
    // Calculate mean
    sum := 0.0
    for _, value := range data {
        sum += value
    }
    mean := sum / float64(len(data))
    
    // Calculate standard deviation
    variance := 0.0
    for _, value := range data {
        variance += math.Pow(value-mean, 2)
    }
    std := math.Sqrt(variance / float64(len(data)))
    
    fe.scalers[featureName] = &StandardScaler{
        Mean:    mean,
        Std:     std,
        Fitted:  true,
    }
}

func (fe *FeatureEngineer) TransformStandardScaler(featureName string, data []float64) []float64 {
    scaler, exists := fe.scalers[featureName]
    if !exists || !scaler.Fitted {
        return data // Return original if not fitted
    }
    
    result := make([]float64, len(data))
    for i, value := range data {
        if scaler.Std == 0 {
            result[i] = 0
        } else {
            result[i] = (value - scaler.Mean) / scaler.Std
        }
    }
    
    return result
}

// Categorical Feature Encoding
func (fe *FeatureEngineer) FitOneHotEncoder(featureName string, data []string) {
    categories := make(map[string]int)
    for _, value := range data {
        if _, exists := categories[value]; !exists {
            categories[value] = len(categories)
        }
    }
    
    fe.encoders[featureName] = &OneHotEncoder{
        Categories: categories,
        Fitted:     true,
    }
}

func (fe *FeatureEngineer) TransformOneHotEncoder(featureName string, data []string) [][]float64 {
    encoder, exists := fe.encoders[featureName]
    if !exists || !encoder.Fitted {
        return nil
    }
    
    numCategories := len(encoder.Categories)
    result := make([][]float64, len(data))
    
    for i, value := range data {
        encoded := make([]float64, numCategories)
        if idx, exists := encoder.Categories[value]; exists {
            encoded[idx] = 1.0
        }
        result[i] = encoded
    }
    
    return result
}

// Feature Selection
func (fe *FeatureEngineer) SelectFeatures(features [][]float64, labels []float64, k int) []int {
    if len(features) == 0 || len(features[0]) == 0 {
        return nil
    }
    
    numFeatures := len(features[0])
    scores := make([]float64, numFeatures)
    
    // Calculate correlation scores
    for i := 0; i < numFeatures; i++ {
        scores[i] = fe.calculateCorrelation(features, labels, i)
    }
    
    // Get top k features
    indices := make([]int, numFeatures)
    for i := range indices {
        indices[i] = i
    }
    
    sort.Slice(indices, func(i, j int) bool {
        return scores[indices[i]] > scores[indices[j]]
    })
    
    if k > len(indices) {
        k = len(indices)
    }
    
    return indices[:k]
}

func (fe *FeatureEngineer) calculateCorrelation(features [][]float64, labels []float64, featureIdx int) float64 {
    if len(features) != len(labels) {
        return 0
    }
    
    n := len(features)
    if n == 0 {
        return 0
    }
    
    // Calculate means
    featureMean := 0.0
    labelMean := 0.0
    for i := 0; i < n; i++ {
        featureMean += features[i][featureIdx]
        labelMean += labels[i]
    }
    featureMean /= float64(n)
    labelMean /= float64(n)
    
    // Calculate correlation
    numerator := 0.0
    featureVar := 0.0
    labelVar := 0.0
    
    for i := 0; i < n; i++ {
        featureDiff := features[i][featureIdx] - featureMean
        labelDiff := labels[i] - labelMean
        
        numerator += featureDiff * labelDiff
        featureVar += featureDiff * featureDiff
        labelVar += labelDiff * labelDiff
    }
    
    if featureVar == 0 || labelVar == 0 {
        return 0
    }
    
    return numerator / math.Sqrt(featureVar*labelVar)
}

// Text Feature Extraction
func (fe *FeatureEngineer) ExtractTextFeatures(texts []string) [][]float64 {
    features := make([][]float64, len(texts))
    
    for i, text := range texts {
        textFeatures := make([]float64, 4)
        
        // Word count
        words := strings.Fields(text)
        textFeatures[0] = float64(len(words))
        
        // Character count
        textFeatures[1] = float64(len(text))
        
        // Average word length
        if len(words) > 0 {
            totalLength := 0
            for _, word := range words {
                totalLength += len(word)
            }
            textFeatures[2] = float64(totalLength) / float64(len(words))
        }
        
        // Special character count
        specialCount := 0
        for _, char := range text {
            if !strings.ContainsRune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ", char) {
                specialCount++
            }
        }
        textFeatures[3] = float64(specialCount)
        
        features[i] = textFeatures
    }
    
    return features
}
```

## Model Deployment

### Containerized Model Deployment

```go
package main

import (
    "context"
    "fmt"
    "log"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"
)

type ModelDeployment struct {
    server     *ModelServer
    httpServer *http.Server
    healthCheck *HealthChecker
    metrics    *MetricsCollector
}

type HealthChecker struct {
    models map[string]bool
    mutex  sync.RWMutex
}

type MetricsCollector struct {
    requestCount    int64
    errorCount      int64
    avgLatency      float64
    totalLatency    int64
    mutex           sync.RWMutex
}

func NewModelDeployment() *ModelDeployment {
    return &ModelDeployment{
        server:      NewModelServer(),
        healthCheck: &HealthChecker{models: make(map[string]bool)},
        metrics:     &MetricsCollector{},
    }
}

func (md *ModelDeployment) DeployModel(modelName string, model MLModel) error {
    // Register model
    md.server.RegisterModel(modelName, model)
    
    // Mark as healthy
    md.healthCheck.mutex.Lock()
    md.healthCheck.models[modelName] = true
    md.healthCheck.mutex.Unlock()
    
    log.Printf("Model %s deployed successfully", modelName)
    return nil
}

func (md *ModelDeployment) StartServer(port string) error {
    mux := http.NewServeMux()
    
    // Prediction endpoint
    mux.HandleFunc("/predict", md.handlePrediction)
    
    // Health check endpoint
    mux.HandleFunc("/health", md.handleHealth)
    
    // Metrics endpoint
    mux.HandleFunc("/metrics", md.handleMetrics)
    
    // Model management endpoints
    mux.HandleFunc("/models", md.handleModels)
    mux.HandleFunc("/models/", md.handleModelOperations)
    
    md.httpServer = &http.Server{
        Addr:    ":" + port,
        Handler: mux,
    }
    
    // Start server in goroutine
    go func() {
        if err := md.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
            log.Fatalf("Server failed to start: %v", err)
        }
    }()
    
    log.Printf("Model server started on port %s", port)
    return nil
}

func (md *ModelDeployment) handlePrediction(w http.ResponseWriter, r *http.Request) {
    start := time.Now()
    
    // Increment request count
    md.metrics.mutex.Lock()
    md.metrics.requestCount++
    md.metrics.mutex.Unlock()
    
    var req PredictionRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        md.incrementErrorCount()
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }
    
    resp, err := md.server.Predict(req)
    if err != nil {
        md.incrementErrorCount()
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    
    // Update latency metrics
    latency := time.Since(start).Milliseconds()
    md.updateLatency(latency)
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(resp)
}

func (md *ModelDeployment) handleHealth(w http.ResponseWriter, r *http.Request) {
    md.healthCheck.mutex.RLock()
    defer md.healthCheck.mutex.RUnlock()
    
    status := "healthy"
    for modelName, healthy := range md.healthCheck.models {
        if !healthy {
            status = "unhealthy"
            break
        }
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]interface{}{
        "status": status,
        "models": md.healthCheck.models,
        "timestamp": time.Now().Unix(),
    })
}

func (md *ModelDeployment) handleMetrics(w http.ResponseWriter, r *http.Request) {
    md.metrics.mutex.RLock()
    defer md.metrics.mutex.RUnlock()
    
    avgLatency := 0.0
    if md.metrics.requestCount > 0 {
        avgLatency = float64(md.metrics.totalLatency) / float64(md.metrics.requestCount)
    }
    
    metrics := map[string]interface{}{
        "request_count": md.metrics.requestCount,
        "error_count":   md.metrics.errorCount,
        "avg_latency_ms": avgLatency,
        "error_rate":    float64(md.metrics.errorCount) / float64(md.metrics.requestCount),
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(metrics)
}

func (md *ModelDeployment) handleModels(w http.ResponseWriter, r *http.Request) {
    if r.Method == http.MethodGet {
        // List models
        md.server.mutex.RLock()
        models := make([]string, 0, len(md.server.models))
        for modelName := range md.server.models {
            models = append(models, modelName)
        }
        md.server.mutex.RUnlock()
        
        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(map[string]interface{}{
            "models": models,
        })
    }
}

func (md *ModelDeployment) handleModelOperations(w http.ResponseWriter, r *http.Request) {
    // Extract model name from path
    path := r.URL.Path
    modelName := strings.TrimPrefix(path, "/models/")
    
    switch r.Method {
    case http.MethodGet:
        // Get model info
        md.server.mutex.RLock()
        _, exists := md.server.models[modelName]
        md.server.mutex.RUnlock()
        
        if !exists {
            http.Error(w, "Model not found", http.StatusNotFound)
            return
        }
        
        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(map[string]interface{}{
            "name": modelName,
            "status": "active",
        })
        
    case http.MethodDelete:
        // Remove model
        md.server.mutex.Lock()
        delete(md.server.models, modelName)
        md.server.mutex.Unlock()
        
        md.healthCheck.mutex.Lock()
        delete(md.healthCheck.models, modelName)
        md.healthCheck.mutex.Unlock()
        
        w.WriteHeader(http.StatusOK)
        json.NewEncoder(w).Encode(map[string]string{
            "message": "Model removed successfully",
        })
    }
}

func (md *ModelDeployment) incrementErrorCount() {
    md.metrics.mutex.Lock()
    md.metrics.errorCount++
    md.metrics.mutex.Unlock()
}

func (md *ModelDeployment) updateLatency(latency int64) {
    md.metrics.mutex.Lock()
    md.metrics.totalLatency += latency
    md.metrics.mutex.Unlock()
}

func (md *ModelDeployment) GracefulShutdown() {
    c := make(chan os.Signal, 1)
    signal.Notify(c, os.Interrupt, syscall.SIGTERM)
    
    <-c
    log.Println("Shutting down server...")
    
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    if err := md.httpServer.Shutdown(ctx); err != nil {
        log.Printf("Server forced to shutdown: %v", err)
    }
    
    log.Println("Server exited")
}
```

## MLOps for Backend

### Model Versioning and A/B Testing

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type ModelVersion struct {
    Version     string
    Model       MLModel
    CreatedAt   time.Time
    Performance map[string]float64
    IsActive    bool
}

type ModelRegistry struct {
    models map[string][]*ModelVersion
    mutex  sync.RWMutex
}

func NewModelRegistry() *ModelRegistry {
    return &ModelRegistry{
        models: make(map[string][]*ModelVersion),
    }
}

func (mr *ModelRegistry) RegisterModel(modelName string, version string, model MLModel) {
    mr.mutex.Lock()
    defer mr.mutex.Unlock()
    
    modelVersion := &ModelVersion{
        Version:   version,
        Model:     model,
        CreatedAt: time.Now(),
        Performance: make(map[string]float64),
        IsActive:  false,
    }
    
    mr.models[modelName] = append(mr.models[modelName], modelVersion)
}

func (mr *ModelRegistry) ActivateModel(modelName string, version string) error {
    mr.mutex.Lock()
    defer mr.mutex.Unlock()
    
    versions, exists := mr.models[modelName]
    if !exists {
        return fmt.Errorf("model not found")
    }
    
    // Deactivate all versions
    for _, v := range versions {
        v.IsActive = false
    }
    
    // Activate specified version
    for _, v := range versions {
        if v.Version == version {
            v.IsActive = true
            return nil
        }
    }
    
    return fmt.Errorf("version not found")
}

func (mr *ModelRegistry) GetActiveModel(modelName string) (MLModel, error) {
    mr.mutex.RLock()
    defer mr.mutex.RUnlock()
    
    versions, exists := mr.models[modelName]
    if !exists {
        return nil, fmt.Errorf("model not found")
    }
    
    for _, v := range versions {
        if v.IsActive {
            return v.Model, nil
        }
    }
    
    return nil, fmt.Errorf("no active model found")
}

// A/B Testing
type ABTest struct {
    TestName     string
    ModelA       string
    ModelB       string
    TrafficSplit float64 // 0.0 to 1.0
    StartTime    time.Time
    EndTime      time.Time
    IsActive     bool
}

type ABTestManager struct {
    tests map[string]*ABTest
    mutex sync.RWMutex
}

func NewABTestManager() *ABTestManager {
    return &ABTestManager{
        tests: make(map[string]*ABTest),
    }
}

func (abm *ABTestManager) CreateTest(testName, modelA, modelB string, trafficSplit float64, duration time.Duration) *ABTest {
    test := &ABTest{
        TestName:     testName,
        ModelA:       modelA,
        ModelB:       modelB,
        TrafficSplit: trafficSplit,
        StartTime:    time.Now(),
        EndTime:      time.Now().Add(duration),
        IsActive:     true,
    }
    
    abm.mutex.Lock()
    abm.tests[testName] = test
    abm.mutex.Unlock()
    
    return test
}

func (abm *ABTestManager) GetModelForRequest(testName string, requestID string) (string, error) {
    abm.mutex.RLock()
    test, exists := abm.tests[testName]
    abm.mutex.RUnlock()
    
    if !exists || !test.IsActive {
        return "", fmt.Errorf("test not found or inactive")
    }
    
    if time.Now().After(test.EndTime) {
        test.IsActive = false
        return "", fmt.Errorf("test has ended")
    }
    
    // Simple hash-based traffic splitting
    hash := hashString(requestID)
    if hash < test.TrafficSplit {
        return test.ModelA, nil
    }
    return test.ModelB, nil
}

func hashString(s string) float64 {
    hash := 0
    for _, c := range s {
        hash = hash*31 + int(c)
    }
    return float64(hash%1000) / 1000.0
}
```

## Interview Questions

### Basic Concepts
1. **How do you serve ML models in production?**
2. **What is the difference between training and inference?**
3. **How do you handle model versioning?**
4. **What are the challenges of ML in backend systems?**
5. **How do you monitor ML model performance?**

### Advanced Topics
1. **How would you implement A/B testing for ML models?**
2. **How do you handle model drift and retraining?**
3. **What are the best practices for ML model deployment?**
4. **How do you implement feature stores?**
5. **How do you handle real-time ML inference?**

### System Design
1. **Design a ML model serving platform.**
2. **How would you implement ML model monitoring?**
3. **Design a feature engineering pipeline.**
4. **How would you implement ML model A/B testing?**
5. **Design a real-time ML inference system.**

## Conclusion

Machine Learning for backend engineers involves understanding how to integrate ML models into production systems. Key areas to master:

- **Model Serving**: APIs, caching, load balancing
- **Feature Engineering**: Data preprocessing, feature selection
- **Model Deployment**: Containerization, orchestration
- **MLOps**: Versioning, monitoring, A/B testing
- **Real-time Systems**: Streaming, low-latency inference
- **Infrastructure**: Scalability, reliability, performance

Understanding these concepts helps in:
- Building ML-powered applications
- Deploying models in production
- Managing ML infrastructure
- Implementing MLOps practices
- Preparing for technical interviews

This guide provides a comprehensive foundation for ML concepts relevant to backend engineers and their practical implementation in Go.


## Real Time Ml Systems

<!-- AUTO-GENERATED ANCHOR: originally referenced as #real-time-ml-systems -->

Placeholder content. Please replace with proper section.


## Ml Infrastructure

<!-- AUTO-GENERATED ANCHOR: originally referenced as #ml-infrastructure -->

Placeholder content. Please replace with proper section.


## Go Implementation Examples

<!-- AUTO-GENERATED ANCHOR: originally referenced as #go-implementation-examples -->

Placeholder content. Please replace with proper section.
