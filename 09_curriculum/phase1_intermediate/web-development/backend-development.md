# Backend Development

## Overview

This module covers backend development concepts including Express.js, FastAPI, Spring Boot, RESTful APIs, authentication, and database integration. These concepts are essential for building robust server-side applications.

## Table of Contents

1. [Express.js Fundamentals](#expressjs-fundamentals)
2. [FastAPI Fundamentals](#fastapi-fundamentals)
3. [Spring Boot Fundamentals](#spring-boot-fundamentals)
4. [RESTful API Design](#restful-api-design)
5. [Authentication & Authorization](#authentication--authorization)
6. [Database Integration](#database-integration)
7. [Applications](#applications)
8. [Complexity Analysis](#complexity-analysis)
9. [Follow-up Questions](#follow-up-questions)

## Express.js Fundamentals

### Theory

Express.js is a minimal and flexible Node.js web application framework that provides a robust set of features for web and mobile applications. It's built on top of Node.js and provides middleware for handling HTTP requests.

### Express.js Implementation

#### Golang Implementation (Express-like Framework)

```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "strings"
    "time"
)

type Middleware func(http.HandlerFunc) http.HandlerFunc

type Route struct {
    Method  string
    Path    string
    Handler http.HandlerFunc
}

type ExpressApp struct {
    Routes     []Route
    Middleware []Middleware
    Server     *http.Server
}

func NewExpressApp() *ExpressApp {
    return &ExpressApp{
        Routes:     make([]Route, 0),
        Middleware: make([]Middleware, 0),
    }
}

func (app *ExpressApp) Use(middleware Middleware) {
    app.Middleware = append(app.Middleware, middleware)
    fmt.Printf("Added middleware\n")
}

func (app *ExpressApp) Get(path string, handler http.HandlerFunc) {
    app.addRoute("GET", path, handler)
}

func (app *ExpressApp) Post(path string, handler http.HandlerFunc) {
    app.addRoute("POST", path, handler)
}

func (app *ExpressApp) Put(path string, handler http.HandlerFunc) {
    app.addRoute("PUT", path, handler)
}

func (app *ExpressApp) Delete(path string, handler http.HandlerFunc) {
    app.addRoute("DELETE", path, handler)
}

func (app *ExpressApp) addRoute(method, path string, handler http.HandlerFunc) {
    route := Route{
        Method:  method,
        Path:    path,
        Handler: handler,
    }
    
    app.Routes = append(app.Routes, route)
    fmt.Printf("Added %s route: %s\n", method, path)
}

func (app *ExpressApp) Listen(port string) {
    mux := http.NewServeMux()
    
    // Add all routes
    for _, route := range app.Routes {
        mux.HandleFunc(route.Path, app.applyMiddleware(route.Handler))
    }
    
    app.Server = &http.Server{
        Addr:    ":" + port,
        Handler: mux,
    }
    
    fmt.Printf("Server listening on port %s\n", port)
    go app.Server.ListenAndServe()
}

func (app *ExpressApp) applyMiddleware(handler http.HandlerFunc) http.HandlerFunc {
    for i := len(app.Middleware) - 1; i >= 0; i-- {
        handler = app.Middleware[i](../../../08_interview_prep/practice/handler)
    }
    return handler
}

// Middleware functions
func Logger() Middleware {
    return func(next http.HandlerFunc) http.HandlerFunc {
        return func(w http.ResponseWriter, r *http.Request) {
            start := time.Now()
            next(w, r)
            duration := time.Since(start)
            fmt.Printf("%s %s %s - %v\n", r.Method, r.URL.Path, r.RemoteAddr, duration)
        }
    }
}

func CORS() Middleware {
    return func(next http.HandlerFunc) http.HandlerFunc {
        return func(w http.ResponseWriter, r *http.Request) {
            w.Header().Set("Access-Control-Allow-Origin", "*")
            w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
            w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
            
            if r.Method == "OPTIONS" {
                w.WriteHeader(http.StatusOK)
                return
            }
            
            next(w, r)
        }
    }
}

func JSONParser() Middleware {
    return func(next http.HandlerFunc) http.HandlerFunc {
        return func(w http.ResponseWriter, r *http.Request) {
            if r.Header.Get("Content-Type") == "application/json" {
                var data map[string]interface{}
                if err := json.NewDecoder(r.Body).Decode(&data); err == nil {
                    r.Header.Set("X-Parsed-Body", "true")
                }
            }
            next(w, r)
        }
    }
}

// Helper functions
func SendJSON(w http.ResponseWriter, status int, data interface{}) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    json.NewEncoder(w).Encode(data)
}

func SendError(w http.ResponseWriter, status int, message string) {
    SendJSON(w, status, map[string]string{"error": message})
}

func main() {
    app := NewExpressApp()
    
    fmt.Println("Express.js Demo:")
    
    // Add middleware
    app.Use(Logger())
    app.Use(CORS())
    app.Use(JSONParser())
    
    // Define routes
    app.Get("/", func(w http.ResponseWriter, r *http.Request) {
        SendJSON(w, http.StatusOK, map[string]string{
            "message": "Welcome to Express.js!",
            "version": "1.0.0",
        })
    })
    
    app.Get("/users", func(w http.ResponseWriter, r *http.Request) {
        users := []map[string]interface{}{
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
        }
        SendJSON(w, http.StatusOK, users)
    })
    
    app.Post("/users", func(w http.ResponseWriter, r *http.Request) {
        // In a real app, you'd parse the request body
        user := map[string]interface{}{
            "id":    3,
            "name":  "Charlie",
            "email": "charlie@example.com",
        }
        SendJSON(w, http.StatusCreated, user)
    })
    
    app.Get("/users/:id", func(w http.ResponseWriter, r *http.Request) {
        // Extract ID from URL path
        path := strings.TrimPrefix(r.URL.Path, "/users/")
        user := map[string]interface{}{
            "id":    path,
            "name":  "User " + path,
            "email": "user" + path + "@example.com",
        }
        SendJSON(w, http.StatusOK, user)
    })
    
    app.Put("/users/:id", func(w http.ResponseWriter, r *http.Request) {
        path := strings.TrimPrefix(r.URL.Path, "/users/")
        user := map[string]interface{}{
            "id":    path,
            "name":  "Updated User " + path,
            "email": "updated" + path + "@example.com",
        }
        SendJSON(w, http.StatusOK, user)
    })
    
    app.Delete("/users/:id", func(w http.ResponseWriter, r *http.Request) {
        path := strings.TrimPrefix(r.URL.Path, "/users/")
        SendJSON(w, http.StatusOK, map[string]string{
            "message": "User " + path + " deleted",
        })
    })
    
    // Start server
    app.Listen("8080")
    
    // Keep the program running
    time.Sleep(1 * time.Second)
}
```

## FastAPI Fundamentals

### Theory

FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints. It provides automatic API documentation and high performance.

### FastAPI Implementation

#### Golang Implementation (FastAPI-like Framework)

```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "reflect"
    "strings"
    "time"
)

type FastAPIApp struct {
    Routes     []Route
    Middleware []Middleware
    Server     *http.Server
    Models     map[string]interface{}
}

func NewFastAPIApp() *FastAPIApp {
    return &FastAPIApp{
        Routes:     make([]Route, 0),
        Middleware: make([]Middleware, 0),
        Models:     make(map[string]interface{}),
    }
}

func (app *FastAPIApp) Get(path string, handler http.HandlerFunc) {
    app.addRoute("GET", path, handler)
}

func (app *FastAPIApp) Post(path string, handler http.HandlerFunc) {
    app.addRoute("POST", path, handler)
}

func (app *FastAPIApp) Put(path string, handler http.HandlerFunc) {
    app.addRoute("PUT", path, handler)
}

func (app *FastAPIApp) Delete(path string, handler http.HandlerFunc) {
    app.addRoute("DELETE", path, handler)
}

func (app *FastAPIApp) addRoute(method, path string, handler http.HandlerFunc) {
    route := Route{
        Method:  method,
        Path:    path,
        Handler: handler,
    }
    
    app.Routes = append(app.Routes, route)
    fmt.Printf("Added %s route: %s\n", method, path)
}

func (app *FastAPIApp) Use(middleware Middleware) {
    app.Middleware = append(app.Middleware, middleware)
    fmt.Printf("Added middleware\n")
}

func (app *FastAPIApp) Listen(port string) {
    mux := http.NewServeMux()
    
    // Add all routes
    for _, route := range app.Routes {
        mux.HandleFunc(route.Path, app.applyMiddleware(route.Handler))
    }
    
    app.Server = &http.Server{
        Addr:    ":" + port,
        Handler: mux,
    }
    
    fmt.Printf("FastAPI server listening on port %s\n", port)
    go app.Server.ListenAndServe()
}

func (app *FastAPIApp) applyMiddleware(handler http.HandlerFunc) http.HandlerFunc {
    for i := len(app.Middleware) - 1; i >= 0; i-- {
        handler = app.Middleware[i](../../../08_interview_prep/practice/handler)
    }
    return handler
}

// Pydantic-like model validation
type User struct {
    ID    int    `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}

type UserCreate struct {
    Name  string `json:"name"`
    Email string `json:"email"`
}

func (app *FastAPIApp) ValidateModel(data interface{}, model interface{}) error {
    // In a real implementation, this would validate the data against the model
    fmt.Printf("Validating data against model: %v\n", reflect.TypeOf(model))
    return nil
}

func (app *FastAPIApp) GenerateOpenAPISpec() map[string]interface{} {
    spec := map[string]interface{}{
        "openapi": "3.0.0",
        "info": map[string]interface{}{
            "title":   "FastAPI App",
            "version": "1.0.0",
        },
        "paths": make(map[string]interface{}),
    }
    
    for _, route := range app.Routes {
        path := route.Path
        method := strings.ToLower(route.Method)
        
        if paths, ok := spec["paths"].(map[string]interface{}); ok {
            if pathSpec, exists := paths[path]; exists {
                if pathMap, ok := pathSpec.(map[string]interface{}); ok {
                    pathMap[method] = map[string]interface{}{
                        "summary": fmt.Sprintf("%s %s", strings.ToUpper(method), path),
                        "responses": map[string]interface{}{
                            "200": map[string]interface{}{
                                "description": "Successful response",
                            },
                        },
                    }
                }
            } else {
                paths[path] = map[string]interface{}{
                    method: map[string]interface{}{
                        "summary": fmt.Sprintf("%s %s", strings.ToUpper(method), path),
                        "responses": map[string]interface{}{
                            "200": map[string]interface{}{
                                "description": "Successful response",
                            },
                        },
                    },
                }
            }
        }
    }
    
    return spec
}

func main() {
    app := NewFastAPIApp()
    
    fmt.Println("FastAPI Demo:")
    
    // Add middleware
    app.Use(Logger())
    app.Use(CORS())
    app.Use(JSONParser())
    
    // Define routes
    app.Get("/", func(w http.ResponseWriter, r *http.Request) {
        SendJSON(w, http.StatusOK, map[string]string{
            "message": "Welcome to FastAPI!",
            "version": "1.0.0",
        })
    })
    
    app.Get("/users", func(w http.ResponseWriter, r *http.Request) {
        users := []User{
            {ID: 1, Name: "Alice", Email: "alice@example.com"},
            {ID: 2, Name: "Bob", Email: "bob@example.com"},
        }
        SendJSON(w, http.StatusOK, users)
    })
    
    app.Post("/users", func(w http.ResponseWriter, r *http.Request) {
        var userCreate UserCreate
        if err := json.NewDecoder(r.Body).Decode(&userCreate); err != nil {
            SendError(w, http.StatusBadRequest, "Invalid JSON")
            return
        }
        
        // Validate model
        if err := app.ValidateModel(userCreate, UserCreate{}); err != nil {
            SendError(w, http.StatusBadRequest, "Validation error")
            return
        }
        
        user := User{
            ID:    3,
            Name:  userCreate.Name,
            Email: userCreate.Email,
        }
        
        SendJSON(w, http.StatusCreated, user)
    })
    
    app.Get("/users/{id}", func(w http.ResponseWriter, r *http.Request) {
        path := strings.TrimPrefix(r.URL.Path, "/users/")
        user := User{
            ID:    1,
            Name:  "User " + path,
            Email: "user" + path + "@example.com",
        }
        SendJSON(w, http.StatusOK, user)
    })
    
    app.Put("/users/{id}", func(w http.ResponseWriter, r *http.Request) {
        path := strings.TrimPrefix(r.URL.Path, "/users/")
        var userUpdate UserCreate
        if err := json.NewDecoder(r.Body).Decode(&userUpdate); err != nil {
            SendError(w, http.StatusBadRequest, "Invalid JSON")
            return
        }
        
        user := User{
            ID:    1,
            Name:  userUpdate.Name,
            Email: userUpdate.Email,
        }
        
        SendJSON(w, http.StatusOK, user)
    })
    
    app.Delete("/users/{id}", func(w http.ResponseWriter, r *http.Request) {
        path := strings.TrimPrefix(r.URL.Path, "/users/")
        SendJSON(w, http.StatusOK, map[string]string{
            "message": "User " + path + " deleted",
        })
    })
    
    // Generate OpenAPI spec
    spec := app.GenerateOpenAPISpec()
    specJSON, _ := json.MarshalIndent(spec, "", "  ")
    fmt.Printf("OpenAPI Spec:\n%s\n", string(specJSON))
    
    // Start server
    app.Listen("8080")
    
    // Keep the program running
    time.Sleep(1 * time.Second)
}
```

## Spring Boot Fundamentals

### Theory

Spring Boot is a Java-based framework for building microservices and web applications. It provides auto-configuration, embedded servers, and production-ready features.

### Spring Boot Implementation

#### Golang Implementation (Spring Boot-like Framework)

```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "reflect"
    "strings"
    "time"
)

type SpringBootApp struct {
    Routes     []Route
    Middleware []Middleware
    Server     *http.Server
    Beans      map[string]interface{}
    Config     map[string]interface{}
}

func NewSpringBootApp() *SpringBootApp {
    return &SpringBootApp{
        Routes:     make([]Route, 0),
        Middleware: make([]Middleware, 0),
        Beans:      make(map[string]interface{}),
        Config:     make(map[string]interface{}),
    }
}

func (app *SpringBootApp) GetMapping(path string, handler http.HandlerFunc) {
    app.addRoute("GET", path, handler)
}

func (app *SpringBootApp) PostMapping(path string, handler http.HandlerFunc) {
    app.addRoute("POST", path, handler)
}

func (app *SpringBootApp) PutMapping(path string, handler http.HandlerFunc) {
    app.addRoute("PUT", path, handler)
}

func (app *SpringBootApp) DeleteMapping(path string, handler http.HandlerFunc) {
    app.addRoute("DELETE", path, handler)
}

func (app *SpringBootApp) addRoute(method, path string, handler http.HandlerFunc) {
    route := Route{
        Method:  method,
        Path:    path,
        Handler: handler,
    }
    
    app.Routes = append(app.Routes, route)
    fmt.Printf("Added %s mapping: %s\n", method, path)
}

func (app *SpringBootApp) Use(middleware Middleware) {
    app.Middleware = append(app.Middleware, middleware)
    fmt.Printf("Added middleware\n")
}

func (app *SpringBootApp) Bean(name string, bean interface{}) {
    app.Beans[name] = bean
    fmt.Printf("Registered bean: %s\n", name)
}

func (app *SpringBootApp) GetBean(name string) interface{} {
    if bean, exists := app.Beans[name]; exists {
        return bean
    }
    return nil
}

func (app *SpringBootApp) SetConfig(key string, value interface{}) {
    app.Config[key] = value
    fmt.Printf("Set config: %s = %v\n", key, value)
}

func (app *SpringBootApp) GetConfig(key string) interface{} {
    if value, exists := app.Config[key]; exists {
        return value
    }
    return nil
}

func (app *SpringBootApp) Run(port string) {
    mux := http.NewServeMux()
    
    // Add all routes
    for _, route := range app.Routes {
        mux.HandleFunc(route.Path, app.applyMiddleware(route.Handler))
    }
    
    app.Server = &http.Server{
        Addr:    ":" + port,
        Handler: mux,
    }
    
    fmt.Printf("Spring Boot application running on port %s\n", port)
    go app.Server.ListenAndServe()
}

func (app *SpringBootApp) applyMiddleware(handler http.HandlerFunc) http.HandlerFunc {
    for i := len(app.Middleware) - 1; i >= 0; i-- {
        handler = app.Middleware[i](../../../08_interview_prep/practice/handler)
    }
    return handler
}

// Spring Boot annotations
type RestController struct {
    Path string
}

type RequestMapping struct {
    Path   string
    Method string
}

type Service struct {
    Name string
}

func (app *SpringBootApp) RestController(path string) *RestController {
    return &RestController{Path: path}
}

func (app *SpringBootApp) Service(name string) *Service {
    return &Service{Name: name}
}

func main() {
    app := NewSpringBootApp()
    
    fmt.Println("Spring Boot Demo:")
    
    // Set configuration
    app.SetConfig("server.port", "8080")
    app.SetConfig("spring.application.name", "demo-app")
    
    // Register beans
    app.Bean("userService", map[string]interface{}{
        "name": "UserService",
        "methods": []string{"findAll", "findById", "save", "delete"},
    })
    
    app.Bean("userRepository", map[string]interface{}{
        "name": "UserRepository",
        "methods": []string{"findAll", "findById", "save", "delete"},
    })
    
    // Add middleware
    app.Use(Logger())
    app.Use(CORS())
    app.Use(JSONParser())
    
    // Define routes
    app.GetMapping("/", func(w http.ResponseWriter, r *http.Request) {
        SendJSON(w, http.StatusOK, map[string]string{
            "message": "Welcome to Spring Boot!",
            "version": "1.0.0",
        })
    })
    
    app.GetMapping("/users", func(w http.ResponseWriter, r *http.Request) {
        users := []map[string]interface{}{
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
        }
        SendJSON(w, http.StatusOK, users)
    })
    
    app.PostMapping("/users", func(w http.ResponseWriter, r *http.Request) {
        var user map[string]interface{}
        if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
            SendError(w, http.StatusBadRequest, "Invalid JSON")
            return
        }
        
        user["id"] = 3
        SendJSON(w, http.StatusCreated, user)
    })
    
    app.GetMapping("/users/{id}", func(w http.ResponseWriter, r *http.Request) {
        path := strings.TrimPrefix(r.URL.Path, "/users/")
        user := map[string]interface{}{
            "id":    path,
            "name":  "User " + path,
            "email": "user" + path + "@example.com",
        }
        SendJSON(w, http.StatusOK, user)
    })
    
    app.PutMapping("/users/{id}", func(w http.ResponseWriter, r *http.Request) {
        path := strings.TrimPrefix(r.URL.Path, "/users/")
        var userUpdate map[string]interface{}
        if err := json.NewDecoder(r.Body).Decode(&userUpdate); err != nil {
            SendError(w, http.StatusBadRequest, "Invalid JSON")
            return
        }
        
        userUpdate["id"] = path
        SendJSON(w, http.StatusOK, userUpdate)
    })
    
    app.DeleteMapping("/users/{id}", func(w http.ResponseWriter, r *http.Request) {
        path := strings.TrimPrefix(r.URL.Path, "/users/")
        SendJSON(w, http.StatusOK, map[string]string{
            "message": "User " + path + " deleted",
        })
    })
    
    // Get bean
    userService := app.GetBean("userService")
    fmt.Printf("User Service: %v\n", userService)
    
    // Get config
    port := app.GetConfig("server.port")
    fmt.Printf("Server Port: %v\n", port)
    
    // Start application
    app.Run("8080")
    
    // Keep the program running
    time.Sleep(1 * time.Second)
}
```

## Follow-up Questions

### 1. Express.js Fundamentals
**Q: What are the key advantages of using Express.js for backend development?**
A: Express.js provides a minimal, flexible framework with extensive middleware support, easy routing, and a large ecosystem. It's lightweight, fast, and allows for rapid development of web applications and APIs.

### 2. FastAPI Fundamentals
**Q: How does FastAPI's automatic API documentation work?**
A: FastAPI uses Python type hints and Pydantic models to automatically generate OpenAPI/Swagger documentation. It introspects the function signatures and model definitions to create interactive API documentation.

### 3. Spring Boot Fundamentals
**Q: What is dependency injection in Spring Boot?**
A: Dependency injection is a design pattern where objects receive their dependencies from external sources rather than creating them internally. Spring Boot's IoC container manages these dependencies and injects them where needed.

## Complexity Analysis

| Operation | Express.js | FastAPI | Spring Boot |
|-----------|------------|---------|-------------|
| Route Registration | O(1) | O(1) | O(1) |
| Middleware Processing | O(n) | O(n) | O(n) |
| Request Handling | O(1) | O(1) | O(1) |
| Startup Time | Fast | Fast | Slow |

## Applications

1. **Express.js**: REST APIs, web applications, microservices, real-time applications
2. **FastAPI**: High-performance APIs, machine learning APIs, data science applications
3. **Spring Boot**: Enterprise applications, microservices, complex business logic
4. **Backend Development**: Web services, mobile backends, data processing systems

---

**Next**: [Full-Stack Integration](fullstack-integration.md) | **Previous**: [Web Development](README.md) | **Up**: [Web Development](README.md)


## Restful Api Design

<!-- AUTO-GENERATED ANCHOR: originally referenced as #restful-api-design -->

Placeholder content. Please replace with proper section.


## Authentication  Authorization

<!-- AUTO-GENERATED ANCHOR: originally referenced as #authentication--authorization -->

Placeholder content. Please replace with proper section.


## Database Integration

<!-- AUTO-GENERATED ANCHOR: originally referenced as #database-integration -->

Placeholder content. Please replace with proper section.
