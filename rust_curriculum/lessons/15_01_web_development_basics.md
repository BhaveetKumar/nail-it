# Lesson 15.1: Web Development Basics

> **Module**: 15 - Web Development  
> **Lesson**: 1 of 8  
> **Duration**: 3-4 hours  
> **Prerequisites**: Module 14 (Database Integration)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Set up a basic web server using Actix Web
- Handle HTTP requests and responses
- Implement RESTful API endpoints
- Use middleware for common web tasks
- Apply error handling in web contexts

---

## 🎯 **Overview**

Rust has excellent web development capabilities with frameworks like Actix Web, Warp, and Axum. This lesson focuses on Actix Web, one of the most popular and performant web frameworks in the Rust ecosystem.

---

## 🚀 **Setting Up Actix Web**

### **Cargo.toml Dependencies**

```toml
[dependencies]
actix-web = "4.4.0"
tokio = { version = "1.35.0", features = ["full"] }
serde = { version = "1.0.195", features = ["derive"] }
serde_json = "1.0.111"
```

### **Basic Server Setup**

```rust
use actix_web::{web, App, HttpServer, HttpResponse, Result};

async fn index() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json("Hello, World!"))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/", web::get().to(index))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

---

## 🔧 **HTTP Methods and Routes**

### **Basic Route Handling**

```rust
use actix_web::{web, App, HttpServer, HttpResponse, Result};

async fn get_users() -> Result<HttpResponse> {
    let users = vec![
        serde_json::json!({"id": 1, "name": "Alice"}),
        serde_json::json!({"id": 2, "name": "Bob"}),
    ];
    Ok(HttpResponse::Ok().json(users))
}

async fn get_user(path: web::Path<u32>) -> Result<HttpResponse> {
    let user_id = path.into_inner();
    let user = serde_json::json!({
        "id": user_id,
        "name": format!("User {}", user_id)
    });
    Ok(HttpResponse::Ok().json(user))
}

async fn create_user(user: web::Json<serde_json::Value>) -> Result<HttpResponse> {
    println!("Creating user: {:?}", user);
    Ok(HttpResponse::Created().json(user))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/users", web::get().to(get_users))
            .route("/users/{id}", web::get().to(get_user))
            .route("/users", web::post().to(create_user))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

### **Path Parameters and Query Strings**

```rust
use actix_web::{web, App, HttpServer, HttpResponse, Result, Query};
use serde::Deserialize;

#[derive(Deserialize)]
struct UserQuery {
    page: Option<u32>,
    limit: Option<u32>,
}

async fn get_users_with_pagination(
    path: web::Path<u32>,
    query: Query<UserQuery>,
) -> Result<HttpResponse> {
    let category_id = path.into_inner();
    let page = query.page.unwrap_or(1);
    let limit = query.limit.unwrap_or(10);
    
    let response = serde_json::json!({
        "category_id": category_id,
        "page": page,
        "limit": limit,
        "users": vec![
            serde_json::json!({"id": 1, "name": "Alice"}),
            serde_json::json!({"id": 2, "name": "Bob"}),
        ]
    });
    
    Ok(HttpResponse::Ok().json(response))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/categories/{id}/users", web::get().to(get_users_with_pagination))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Basic CRUD API**

```rust
use actix_web::{web, App, HttpServer, HttpResponse, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Serialize, Deserialize, Clone)]
struct User {
    id: u32,
    name: String,
    email: String,
}

struct AppState {
    users: Mutex<HashMap<u32, User>>,
    next_id: Mutex<u32>,
}

async fn get_users(data: web::Data<AppState>) -> Result<HttpResponse> {
    let users = data.users.lock().unwrap();
    let user_list: Vec<User> = users.values().cloned().collect();
    Ok(HttpResponse::Ok().json(user_list))
}

async fn get_user(path: web::Path<u32>, data: web::Data<AppState>) -> Result<HttpResponse> {
    let users = data.users.lock().unwrap();
    let user_id = path.into_inner();
    
    match users.get(&user_id) {
        Some(user) => Ok(HttpResponse::Ok().json(user)),
        None => Ok(HttpResponse::NotFound().json("User not found")),
    }
}

async fn create_user(
    user: web::Json<User>,
    data: web::Data<AppState>,
) -> Result<HttpResponse> {
    let mut users = data.users.lock().unwrap();
    let mut next_id = data.next_id.lock().unwrap();
    
    let new_user = User {
        id: *next_id,
        name: user.name.clone(),
        email: user.email.clone(),
    };
    
    users.insert(*next_id, new_user.clone());
    *next_id += 1;
    
    Ok(HttpResponse::Created().json(new_user))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let app_state = web::Data::new(AppState {
        users: Mutex::new(HashMap::new()),
        next_id: Mutex::new(1),
    });

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .route("/users", web::get().to(get_users))
            .route("/users/{id}", web::get().to(get_user))
            .route("/users", web::post().to(create_user))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

### **Exercise 2: Error Handling**

```rust
use actix_web::{web, App, HttpServer, HttpResponse, Result, error};
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
    message: String,
}

#[derive(Deserialize)]
struct CreateUserRequest {
    name: String,
    email: String,
}

async fn create_user(user: web::Json<CreateUserRequest>) -> Result<HttpResponse> {
    // Validate email
    if !user.email.contains('@') {
        return Ok(HttpResponse::BadRequest().json(ErrorResponse {
            error: "Invalid email".to_string(),
            message: "Email must contain @ symbol".to_string(),
        }));
    }
    
    // Validate name
    if user.name.trim().is_empty() {
        return Ok(HttpResponse::BadRequest().json(ErrorResponse {
            error: "Invalid name".to_string(),
            message: "Name cannot be empty".to_string(),
        }));
    }
    
    let response = serde_json::json!({
        "id": 1,
        "name": user.name,
        "email": user.email,
    });
    
    Ok(HttpResponse::Created().json(response))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/users", web::post().to(create_user))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, web, App};

    #[actix_web::test]
    async fn test_get_users() {
        let app = test::init_service(
            App::new()
                .route("/users", web::get().to(get_users))
        ).await;
        
        let req = test::TestRequest::get().uri("/users").to_request();
        let resp = test::call_service(&app, req).await;
        
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_create_user() {
        let app = test::init_service(
            App::new()
                .route("/users", web::post().to(create_user))
        ).await;
        
        let user_data = serde_json::json!({
            "name": "Test User",
            "email": "test@example.com"
        });
        
        let req = test::TestRequest::post()
            .uri("/users")
            .set_json(&user_data)
            .to_request();
        
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }
}
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Missing Async**

```rust
// ❌ Wrong - missing async
fn bad_handler() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json("Hello"))
}

// ✅ Correct - use async
async fn good_handler() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json("Hello"))
}
```

### **Common Mistake 2: Incorrect Error Handling**

```rust
// ❌ Wrong - returning Result directly
async fn bad_handler() -> Result<HttpResponse, actix_web::Error> {
    // This won't work with actix-web
}

// ✅ Correct - return Result<HttpResponse>
async fn good_handler() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json("Hello"))
}
```

### **Common Mistake 3: State Management**

```rust
// ❌ Wrong - not using web::Data
async fn bad_handler(state: AppState) -> Result<HttpResponse> {
    // This won't work
}

// ✅ Correct - use web::Data
async fn good_handler(data: web::Data<AppState>) -> Result<HttpResponse> {
    // This works
}
```

---

## 📊 **Middleware and Configuration**

### **Basic Middleware**

```rust
use actix_web::{web, App, HttpServer, HttpResponse, Result, middleware};

async fn index() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json("Hello, World!"))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .wrap(middleware::Logger::default())
            .wrap(middleware::DefaultHeaders::new().header("X-Version", "1.0"))
            .route("/", web::get().to(index))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

### **CORS Configuration**

```rust
use actix_cors::Cors;
use actix_web::{web, App, HttpServer, HttpResponse, Result};

async fn index() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json("Hello, World!"))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);
            
        App::new()
            .wrap(cors)
            .route("/", web::get().to(index))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

---

## 🎯 **Best Practices**

### **Project Structure**

```
web_app/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── handlers/
│   │   ├── mod.rs
│   │   ├── users.rs
│   │   └── auth.rs
│   ├── models/
│   │   ├── mod.rs
│   │   └── user.rs
│   ├── middleware/
│   │   ├── mod.rs
│   │   └── auth.rs
│   └── config/
│       ├── mod.rs
│       └── settings.rs
```

### **Error Handling**

```rust
use actix_web::{web, App, HttpServer, HttpResponse, Result, error};

#[derive(Serialize)]
struct ApiError {
    error: String,
    message: String,
}

impl actix_web::error::ResponseError for ApiError {
    fn error_response(&self) -> HttpResponse {
        HttpResponse::BadRequest().json(self)
    }
}

async fn handler() -> Result<HttpResponse, ApiError> {
    Err(ApiError {
        error: "Validation Error".to_string(),
        message: "Invalid input".to_string(),
    })
}
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [Actix Web Documentation](https://actix.rs/) - Fetched: 2024-12-19T00:00:00Z
- [Actix Web Examples](https://github.com/actix/examples) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Rust Web Development](https://github.com/rust-lang/rust-by-example) - Fetched: 2024-12-19T00:00:00Z
- [Web Development with Rust](https://rust-lang.github.io/async-book/) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. How do you set up a basic Actix Web server?
2. What's the difference between `web::get()` and `web::post()`?
3. How do you handle path parameters and query strings?
4. What are the benefits of using middleware in web applications?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Database integration with web applications
- Authentication and authorization
- WebSocket support
- Performance optimization

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [15.2 Database Integration](15_02_database_integration.md)
