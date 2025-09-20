# RESTful APIs

## Overview

This module covers RESTful API design concepts including HTTP methods, status codes, resource design, versioning, and best practices. These concepts are essential for building well-designed, maintainable APIs.

## Table of Contents

1. [HTTP Methods & Status Codes](#http-methods--status-codes)
2. [Resource Design](#resource-design)
3. [API Versioning](#api-versioning)
4. [Error Handling](#error-handling)
5. [Pagination & Filtering](#pagination--filtering)
6. [Applications](#applications)
7. [Complexity Analysis](#complexity-analysis)
8. [Follow-up Questions](#follow-up-questions)

## HTTP Methods & Status Codes

### Theory

RESTful APIs use HTTP methods to perform operations on resources and HTTP status codes to indicate the result of operations. Understanding these concepts is crucial for building consistent APIs.

### HTTP Method Implementation

#### Golang Implementation

```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "strconv"
    "strings"
    "time"
)

type HTTPMethod string

const (
    GET    HTTPMethod = "GET"
    POST   HTTPMethod = "POST"
    PUT    HTTPMethod = "PUT"
    PATCH  HTTPMethod = "PATCH"
    DELETE HTTPMethod = "DELETE"
    HEAD   HTTPMethod = "HEAD"
    OPTIONS HTTPMethod = "OPTIONS"
)

type StatusCode int

const (
    StatusOK                  StatusCode = 200
    StatusCreated             StatusCode = 201
    StatusAccepted            StatusCode = 202
    StatusNoContent           StatusCode = 204
    StatusBadRequest          StatusCode = 400
    StatusUnauthorized        StatusCode = 401
    StatusForbidden           StatusCode = 403
    StatusNotFound            StatusCode = 404
    StatusMethodNotAllowed    StatusCode = 405
    StatusConflict            StatusCode = 409
    StatusUnprocessableEntity StatusCode = 422
    StatusTooManyRequests     StatusCode = 429
    StatusInternalServerError StatusCode = 500
    StatusNotImplemented      StatusCode = 501
    StatusBadGateway          StatusCode = 502
    StatusServiceUnavailable  StatusCode = 503
)

type APIResponse struct {
    Status  StatusCode `json:"status"`
    Message string     `json:"message"`
    Data    interface{} `json:"data,omitempty"`
    Error   string     `json:"error,omitempty"`
    Meta    map[string]interface{} `json:"meta,omitempty"`
}

type RESTfulAPI struct {
    Routes map[string]map[HTTPMethod]http.HandlerFunc
    mutex  sync.RWMutex
}

func NewRESTfulAPI() *RESTfulAPI {
    return &RESTfulAPI{
        Routes: make(map[string]map[HTTPMethod]http.HandlerFunc),
    }
}

func (api *RESTfulAPI) AddRoute(path string, method HTTPMethod, handler http.HandlerFunc) {
    api.mutex.Lock()
    defer api.mutex.Unlock()
    
    if api.Routes[path] == nil {
        api.Routes[path] = make(map[HTTPMethod]http.HandlerFunc)
    }
    
    api.Routes[path][method] = handler
    fmt.Printf("Added route: %s %s\n", method, path)
}

func (api *RESTfulAPI) HandleRequest(w http.ResponseWriter, r *http.Request) {
    method := HTTPMethod(r.Method)
    path := r.URL.Path
    
    api.mutex.RLock()
    route, exists := api.Routes[path]
    api.mutex.RUnlock()
    
    if !exists {
        api.SendResponse(w, StatusNotFound, "Route not found", nil)
        return
    }
    
    handler, exists := route[method]
    if !exists {
        api.SendResponse(w, StatusMethodNotAllowed, "Method not allowed", nil)
        return
    }
    
    handler(w, r)
}

func (api *RESTfulAPI) SendResponse(w http.ResponseWriter, status StatusCode, message string, data interface{}) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(int(status))
    
    response := APIResponse{
        Status:  status,
        Message: message,
        Data:    data,
    }
    
    json.NewEncoder(w).Encode(response)
}

func (api *RESTfulAPI) SendError(w http.ResponseWriter, status StatusCode, error string) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(int(status))
    
    response := APIResponse{
        Status: status,
        Error:  error,
    }
    
    json.NewEncoder(w).Encode(response)
}

func (api *RESTfulAPI) SendSuccess(w http.ResponseWriter, status StatusCode, message string, data interface{}) {
    api.SendResponse(w, status, message, data)
}

func (api *RESTfulAPI) Start(port string) {
    mux := http.NewServeMux()
    mux.HandleFunc("/", api.HandleRequest)
    
    server := &http.Server{
        Addr:    ":" + port,
        Handler: mux,
    }
    
    fmt.Printf("RESTful API server starting on port %s\n", port)
    go server.ListenAndServe()
}

func main() {
    api := NewRESTfulAPI()
    
    fmt.Println("RESTful API Demo:")
    
    // Add routes
    api.AddRoute("/users", GET, func(w http.ResponseWriter, r *http.Request) {
        users := []map[string]interface{}{
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
        }
        api.SendSuccess(w, StatusOK, "Users retrieved successfully", users)
    })
    
    api.AddRoute("/users", POST, func(w http.ResponseWriter, r *http.Request) {
        var user map[string]interface{}
        if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
            api.SendError(w, StatusBadRequest, "Invalid JSON")
            return
        }
        
        user["id"] = 3
        user["created_at"] = time.Now().Format(time.RFC3339)
        
        api.SendSuccess(w, StatusCreated, "User created successfully", user)
    })
    
    api.AddRoute("/users/{id}", GET, func(w http.ResponseWriter, r *http.Request) {
        // Extract ID from URL path
        path := strings.TrimPrefix(r.URL.Path, "/users/")
        id, err := strconv.Atoi(path)
        if err != nil {
            api.SendError(w, StatusBadRequest, "Invalid user ID")
            return
        }
        
        user := map[string]interface{}{
            "id":    id,
            "name":  "User " + path,
            "email": "user" + path + "@example.com",
        }
        
        api.SendSuccess(w, StatusOK, "User retrieved successfully", user)
    })
    
    api.AddRoute("/users/{id}", PUT, func(w http.ResponseWriter, r *http.Request) {
        path := strings.TrimPrefix(r.URL.Path, "/users/")
        id, err := strconv.Atoi(path)
        if err != nil {
            api.SendError(w, StatusBadRequest, "Invalid user ID")
            return
        }
        
        var userUpdate map[string]interface{}
        if err := json.NewDecoder(r.Body).Decode(&userUpdate); err != nil {
            api.SendError(w, StatusBadRequest, "Invalid JSON")
            return
        }
        
        userUpdate["id"] = id
        userUpdate["updated_at"] = time.Now().Format(time.RFC3339)
        
        api.SendSuccess(w, StatusOK, "User updated successfully", userUpdate)
    })
    
    api.AddRoute("/users/{id}", PATCH, func(w http.ResponseWriter, r *http.Request) {
        path := strings.TrimPrefix(r.URL.Path, "/users/")
        id, err := strconv.Atoi(path)
        if err != nil {
            api.SendError(w, StatusBadRequest, "Invalid user ID")
            return
        }
        
        var userUpdate map[string]interface{}
        if err := json.NewDecoder(r.Body).Decode(&userUpdate); err != nil {
            api.SendError(w, StatusBadRequest, "Invalid JSON")
            return
        }
        
        userUpdate["id"] = id
        userUpdate["updated_at"] = time.Now().Format(time.RFC3339)
        
        api.SendSuccess(w, StatusOK, "User partially updated successfully", userUpdate)
    })
    
    api.AddRoute("/users/{id}", DELETE, func(w http.ResponseWriter, r *http.Request) {
        path := strings.TrimPrefix(r.URL.Path, "/users/")
        id, err := strconv.Atoi(path)
        if err != nil {
            api.SendError(w, StatusBadRequest, "Invalid user ID")
            return
        }
        
        api.SendSuccess(w, StatusNoContent, "User deleted successfully", nil)
    })
    
    // Start server
    api.Start("8080")
    
    // Keep the program running
    time.Sleep(1 * time.Second)
}
```

## Resource Design

### Theory

Resource design is the foundation of RESTful APIs. Resources should be nouns, have clear relationships, and follow consistent naming conventions. Good resource design makes APIs intuitive and easy to use.

### Resource Design Implementation

#### Golang Implementation

```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "strconv"
    "strings"
    "time"
)

type Resource struct {
    ID        int                    `json:"id"`
    Type      string                 `json:"type"`
    Attributes map[string]interface{} `json:"attributes"`
    Links     map[string]string      `json:"links,omitempty"`
    Meta      map[string]interface{} `json:"meta,omitempty"`
}

type ResourceCollection struct {
    Data  []Resource              `json:"data"`
    Links map[string]string       `json:"links,omitempty"`
    Meta  map[string]interface{}  `json:"meta,omitempty"`
}

type ResourceManager struct {
    Resources map[string]map[int]*Resource
    mutex     sync.RWMutex
}

func NewResourceManager() *ResourceManager {
    return &ResourceManager{
        Resources: make(map[string]map[int]*Resource),
    }
}

func (rm *ResourceManager) CreateResource(resourceType string, attributes map[string]interface{}) *Resource {
    rm.mutex.Lock()
    defer rm.mutex.Unlock()
    
    if rm.Resources[resourceType] == nil {
        rm.Resources[resourceType] = make(map[int]*Resource)
    }
    
    id := len(rm.Resources[resourceType]) + 1
    resource := &Resource{
        ID:         id,
        Type:       resourceType,
        Attributes: attributes,
        Links:      make(map[string]string),
        Meta:       make(map[string]interface{}),
    }
    
    // Add self link
    resource.Links["self"] = fmt.Sprintf("/%s/%d", resourceType, id)
    
    // Add meta information
    resource.Meta["created_at"] = time.Now().Format(time.RFC3339)
    resource.Meta["updated_at"] = time.Now().Format(time.RFC3339)
    
    rm.Resources[resourceType][id] = resource
    
    fmt.Printf("Created resource: %s/%d\n", resourceType, id)
    return resource
}

func (rm *ResourceManager) GetResource(resourceType string, id int) *Resource {
    rm.mutex.RLock()
    defer rm.mutex.RUnlock()
    
    if resources, exists := rm.Resources[resourceType]; exists {
        if resource, exists := resources[id]; exists {
            return resource
        }
    }
    
    return nil
}

func (rm *ResourceManager) UpdateResource(resourceType string, id int, attributes map[string]interface{}) *Resource {
    rm.mutex.Lock()
    defer rm.mutex.Unlock()
    
    if resources, exists := rm.Resources[resourceType]; exists {
        if resource, exists := resources[id]; exists {
            // Update attributes
            for key, value := range attributes {
                resource.Attributes[key] = value
            }
            
            // Update meta
            resource.Meta["updated_at"] = time.Now().Format(time.RFC3339)
            
            fmt.Printf("Updated resource: %s/%d\n", resourceType, id)
            return resource
        }
    }
    
    return nil
}

func (rm *ResourceManager) DeleteResource(resourceType string, id int) bool {
    rm.mutex.Lock()
    defer rm.mutex.Unlock()
    
    if resources, exists := rm.Resources[resourceType]; exists {
        if _, exists := resources[id]; exists {
            delete(resources, id)
            fmt.Printf("Deleted resource: %s/%d\n", resourceType, id)
            return true
        }
    }
    
    return false
}

func (rm *ResourceManager) ListResources(resourceType string, limit, offset int) *ResourceCollection {
    rm.mutex.RLock()
    defer rm.mutex.RUnlock()
    
    if resources, exists := rm.Resources[resourceType]; exists {
        var collection []Resource
        
        count := 0
        for _, resource := range resources {
            if count >= offset {
                collection = append(collection, *resource)
                if len(collection) >= limit {
                    break
                }
            }
            count++
        }
        
        return &ResourceCollection{
            Data: collection,
            Links: map[string]string{
                "self": fmt.Sprintf("/%s?limit=%d&offset=%d", resourceType, limit, offset),
            },
            Meta: map[string]interface{}{
                "total":    len(resources),
                "limit":    limit,
                "offset":   offset,
                "count":    len(collection),
            },
        }
    }
    
    return &ResourceCollection{
        Data:  []Resource{},
        Links: make(map[string]string),
        Meta:  make(map[string]interface{}),
    }
}

func (rm *ResourceManager) AddRelationship(resourceType string, id int, relationshipType, relatedType string, relatedID int) {
    rm.mutex.Lock()
    defer rm.mutex.Unlock()
    
    if resources, exists := rm.Resources[resourceType]; exists {
        if resource, exists := resources[id]; exists {
            if resource.Attributes["relationships"] == nil {
                resource.Attributes["relationships"] = make(map[string]interface{})
            }
            
            relationships := resource.Attributes["relationships"].(map[string]interface{})
            relationships[relationshipType] = map[string]interface{}{
                "type": relatedType,
                "id":   relatedID,
            }
            
            fmt.Printf("Added relationship: %s/%d -> %s/%d\n", resourceType, id, relatedType, relatedID)
        }
    }
}

func main() {
    rm := NewResourceManager()
    
    fmt.Println("Resource Design Demo:")
    
    // Create users
    user1 := rm.CreateResource("users", map[string]interface{}{
        "name":  "Alice",
        "email": "alice@example.com",
        "age":   25,
    })
    
    user2 := rm.CreateResource("users", map[string]interface{}{
        "name":  "Bob",
        "email": "bob@example.com",
        "age":   30,
    })
    
    // Create posts
    post1 := rm.CreateResource("posts", map[string]interface{}{
        "title":   "Hello World",
        "content": "This is my first post",
        "author":  user1.ID,
    })
    
    post2 := rm.CreateResource("posts", map[string]interface{}{
        "title":   "Second Post",
        "content": "This is my second post",
        "author":  user2.ID,
    })
    
    // Add relationships
    rm.AddRelationship("users", user1.ID, "posts", "posts", post1.ID)
    rm.AddRelationship("users", user2.ID, "posts", "posts", post2.ID)
    
    // List resources
    users := rm.ListResources("users", 10, 0)
    fmt.Printf("Users: %d total, %d returned\n", users.Meta["total"], users.Meta["count"])
    
    posts := rm.ListResources("posts", 10, 0)
    fmt.Printf("Posts: %d total, %d returned\n", posts.Meta["total"], posts.Meta["count"])
    
    // Get specific resource
    user := rm.GetResource("users", 1)
    if user != nil {
        fmt.Printf("User 1: %s (%s)\n", user.Attributes["name"], user.Attributes["email"])
    }
    
    // Update resource
    rm.UpdateResource("users", 1, map[string]interface{}{
        "age": 26,
    })
    
    // Get updated resource
    user = rm.GetResource("users", 1)
    if user != nil {
        fmt.Printf("Updated User 1: age %v\n", user.Attributes["age"])
    }
    
    // Delete resource
    rm.DeleteResource("posts", 2)
    
    // List posts after deletion
    posts = rm.ListResources("posts", 10, 0)
    fmt.Printf("Posts after deletion: %d total\n", posts.Meta["total"])
}
```

## API Versioning

### Theory

API versioning allows you to evolve your API while maintaining backward compatibility. Common versioning strategies include URL versioning, header versioning, and query parameter versioning.

### API Versioning Implementation

#### Golang Implementation

```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "strconv"
    "strings"
    "time"
)

type APIVersion string

const (
    V1 APIVersion = "v1"
    V2 APIVersion = "v2"
    V3 APIVersion = "v3"
)

type VersionedAPI struct {
    Versions map[APIVersion]map[string]map[HTTPMethod]http.HandlerFunc
    mutex    sync.RWMutex
}

func NewVersionedAPI() *VersionedAPI {
    return &VersionedAPI{
        Versions: make(map[APIVersion]map[string]map[HTTPMethod]http.HandlerFunc),
    }
}

func (api *VersionedAPI) AddRoute(version APIVersion, path string, method HTTPMethod, handler http.HandlerFunc) {
    api.mutex.Lock()
    defer api.mutex.Unlock()
    
    if api.Versions[version] == nil {
        api.Versions[version] = make(map[string]map[HTTPMethod]http.HandlerFunc)
    }
    
    if api.Versions[version][path] == nil {
        api.Versions[version][path] = make(map[HTTPMethod]http.HandlerFunc)
    }
    
    api.Versions[version][path][method] = handler
    fmt.Printf("Added route: %s %s %s\n", version, method, path)
}

func (api *VersionedAPI) HandleRequest(w http.ResponseWriter, r *http.Request) {
    // Extract version from URL path
    pathParts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
    if len(pathParts) == 0 {
        api.SendError(w, StatusBadRequest, "Invalid path")
        return
    }
    
    version := APIVersion(pathParts[0])
    if !api.IsValidVersion(version) {
        api.SendError(w, StatusBadRequest, "Invalid API version")
        return
    }
    
    // Remove version from path
    actualPath := "/" + strings.Join(pathParts[1:], "/")
    if actualPath == "/" {
        actualPath = "/"
    }
    
    method := HTTPMethod(r.Method)
    
    api.mutex.RLock()
    versionRoutes, exists := api.Versions[version]
    api.mutex.RUnlock()
    
    if !exists {
        api.SendError(w, StatusNotFound, "Version not found")
        return
    }
    
    route, exists := versionRoutes[actualPath]
    if !exists {
        api.SendError(w, StatusNotFound, "Route not found")
        return
    }
    
    handler, exists := route[method]
    if !exists {
        api.SendError(w, StatusMethodNotAllowed, "Method not allowed")
        return
    }
    
    // Add version to request context
    r.Header.Set("X-API-Version", string(version))
    
    handler(w, r)
}

func (api *VersionedAPI) IsValidVersion(version APIVersion) bool {
    validVersions := []APIVersion{V1, V2, V3}
    for _, v := range validVersions {
        if v == version {
            return true
        }
    }
    return false
}

func (api *VersionedAPI) SendError(w http.ResponseWriter, status StatusCode, message string) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(int(status))
    
    response := map[string]interface{}{
        "error":   message,
        "status":  int(status),
        "version": "unknown",
    }
    
    json.NewEncoder(w).Encode(response)
}

func (api *VersionedAPI) SendResponse(w http.ResponseWriter, status StatusCode, message string, data interface{}) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(int(status))
    
    response := map[string]interface{}{
        "status":  int(status),
        "message": message,
        "data":    data,
    }
    
    json.NewEncoder(w).Encode(response)
}

func (api *VersionedAPI) Start(port string) {
    mux := http.NewServeMux()
    mux.HandleFunc("/", api.HandleRequest)
    
    server := &http.Server{
        Addr:    ":" + port,
        Handler: mux,
    }
    
    fmt.Printf("Versioned API server starting on port %s\n", port)
    go server.ListenAndServe()
}

func main() {
    api := NewVersionedAPI()
    
    fmt.Println("API Versioning Demo:")
    
    // V1 routes
    api.AddRoute(V1, "/users", GET, func(w http.ResponseWriter, r *http.Request) {
        users := []map[string]interface{}{
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
        }
        api.SendResponse(w, StatusOK, "Users retrieved (V1)", users)
    })
    
    api.AddRoute(V1, "/users", POST, func(w http.ResponseWriter, r *http.Request) {
        var user map[string]interface{}
        if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
            api.SendError(w, StatusBadRequest, "Invalid JSON")
            return
        }
        
        user["id"] = 1
        user["version"] = "v1"
        api.SendResponse(w, StatusCreated, "User created (V1)", user)
    })
    
    // V2 routes
    api.AddRoute(V2, "/users", GET, func(w http.ResponseWriter, r *http.Request) {
        users := []map[string]interface{}{
            {"id": 1, "name": "Alice", "email": "alice@example.com", "role": "user"},
        }
        api.SendResponse(w, StatusOK, "Users retrieved (V2)", users)
    })
    
    api.AddRoute(V2, "/users", POST, func(w http.ResponseWriter, r *http.Request) {
        var user map[string]interface{}
        if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
            api.SendError(w, StatusBadRequest, "Invalid JSON")
            return
        }
        
        user["id"] = 1
        user["version"] = "v2"
        user["role"] = "user"
        api.SendResponse(w, StatusCreated, "User created (V2)", user)
    })
    
    // V3 routes
    api.AddRoute(V3, "/users", GET, func(w http.ResponseWriter, r *http.Request) {
        users := []map[string]interface{}{
            {"id": 1, "name": "Alice", "email": "alice@example.com", "role": "user", "permissions": []string{"read", "write"}},
        }
        api.SendResponse(w, StatusOK, "Users retrieved (V3)", users)
    })
    
    api.AddRoute(V3, "/users", POST, func(w http.ResponseWriter, r *http.Request) {
        var user map[string]interface{}
        if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
            api.SendError(w, StatusBadRequest, "Invalid JSON")
            return
        }
        
        user["id"] = 1
        user["version"] = "v3"
        user["role"] = "user"
        user["permissions"] = []string{"read", "write"}
        api.SendResponse(w, StatusCreated, "User created (V3)", user)
    })
    
    // Start server
    api.Start("8080")
    
    // Keep the program running
    time.Sleep(1 * time.Second)
}
```

## Follow-up Questions

### 1. HTTP Methods & Status Codes
**Q: When should you use PUT vs PATCH for updating resources?**
A: Use PUT when you want to replace the entire resource with the provided data. Use PATCH when you want to partially update specific fields of the resource.

### 2. Resource Design
**Q: What are the key principles of good resource design?**
A: Use nouns for resource names, make resources hierarchical, use consistent naming conventions, include proper relationships, and design for discoverability.

### 3. API Versioning
**Q: What are the trade-offs between different API versioning strategies?**
A: URL versioning is simple but clutters URLs. Header versioning keeps URLs clean but requires client changes. Query parameter versioning is flexible but can be confusing.

## Complexity Analysis

| Operation | HTTP Methods | Resource Design | API Versioning |
|-----------|--------------|-----------------|----------------|
| Route Registration | O(1) | O(1) | O(1) |
| Request Processing | O(1) | O(1) | O(1) |
| Resource Creation | N/A | O(1) | N/A |
| Version Resolution | N/A | N/A | O(1) |

## Applications

1. **HTTP Methods**: RESTful APIs, web services, microservices
2. **Resource Design**: API design, data modeling, system architecture
3. **API Versioning**: API evolution, backward compatibility, client management
4. **Error Handling**: User experience, debugging, monitoring

---

**Next**: [GraphQL APIs](graphql-apis.md) | **Previous**: [API Design](README.md) | **Up**: [API Design](README.md)
