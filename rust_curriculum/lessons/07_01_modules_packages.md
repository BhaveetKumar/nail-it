---
# Auto-generated front matter
Title: 07 01 Modules Packages
LastUpdated: 2025-11-06T20:45:58.126632
Tags: []
Status: draft
---

# Lesson 7.1: Modules and Packages

> **Module**: 7 - Modules and Crates  
> **Lesson**: 1 of 6  
> **Duration**: 2-3 hours  
> **Prerequisites**: Module 6 (Collections and Iterators)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Organize code using modules and packages
- Understand the module system and visibility rules
- Create and use crates effectively
- Manage dependencies and publish packages
- Apply best practices for code organization

---

## 🎯 **Overview**

Rust's module system helps you organize code into logical units and control privacy. Modules allow you to group related functionality together and control what parts of your code are public or private.

---

## 🏗️ **Basic Module System**

### **Creating Modules**

```rust
// In main.rs or lib.rs
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {
            println!("Adding to waitlist");
        }
        
        fn seat_at_table() {
            println!("Seating at table");
        }
    }
    
    mod serving {
        fn take_order() {
            println!("Taking order");
        }
        
        fn serve_order() {
            println!("Serving order");
        }
        
        fn take_payment() {
            println!("Taking payment");
        }
    }
}

pub fn eat_at_restaurant() {
    // Absolute path
    crate::front_of_house::hosting::add_to_waitlist();
    
    // Relative path
    front_of_house::hosting::add_to_waitlist();
}

fn main() {
    eat_at_restaurant();
}
```

### **Module Visibility**

```rust
mod outer {
    pub mod inner {
        pub fn public_function() {
            println!("This is public");
        }
        
        fn private_function() {
            println!("This is private");
        }
        
        pub fn indirect_access() {
            println!("Calling private function:");
            private_function();
        }
    }
}

fn main() {
    outer::inner::public_function();
    outer::inner::indirect_access();
    // outer::inner::private_function(); // Error: private function
}
```

---

## 📁 **File-based Modules**

### **Project Structure**

```
restaurant/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── front_of_house.rs
│   └── back_of_house.rs
```

### **Module Files**

```rust
// src/front_of_house.rs
pub mod hosting {
    pub fn add_to_waitlist() {
        println!("Adding to waitlist");
    }
}

pub mod serving {
    pub fn take_order() {
        println!("Taking order");
    }
}
```

```rust
// src/back_of_house.rs
pub struct Breakfast {
    pub toast: String,
    seasonal_fruit: String,
}

impl Breakfast {
    pub fn summer(toast: &str) -> Breakfast {
        Breakfast {
            toast: String::from(toast),
            seasonal_fruit: String::from("peaches"),
        }
    }
}

pub enum Appetizer {
    Soup,
    Salad,
}
```

```rust
// src/lib.rs
pub mod front_of_house;
pub mod back_of_house;

pub use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    // Using re-exported module
    hosting::add_to_waitlist();
    
    // Using struct
    let mut meal = back_of_house::Breakfast::summer("Rye");
    meal.toast = String::from("Wheat");
    println!("I'd like {} toast please", meal.toast);
    
    // Using enum
    let order1 = back_of_house::Appetizer::Soup;
    let order2 = back_of_house::Appetizer::Salad;
}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Basic Module Structure**

```rust
// src/math.rs
pub mod basic {
    pub fn add(a: i32, b: i32) -> i32 {
        a + b
    }
    
    pub fn multiply(a: i32, b: i32) -> i32 {
        a * b
    }
}

pub mod advanced {
    pub fn power(base: i32, exponent: u32) -> i32 {
        base.pow(exponent)
    }
    
    pub fn factorial(n: u32) -> u32 {
        if n <= 1 {
            1
        } else {
            n * factorial(n - 1)
        }
    }
}

// src/lib.rs
pub mod math;

pub use math::basic;

fn main() {
    println!("2 + 3 = {}", basic::add(2, 3));
    println!("2 * 3 = {}", basic::multiply(2, 3));
    println!("2^3 = {}", math::advanced::power(2, 3));
    println!("5! = {}", math::advanced::factorial(5));
}
```

### **Exercise 2: Library Crate**

```rust
// src/lib.rs
pub mod geometry {
    pub mod circle {
        use std::f64::consts::PI;
        
        pub struct Circle {
            radius: f64,
        }
        
        impl Circle {
            pub fn new(radius: f64) -> Circle {
                Circle { radius }
            }
            
            pub fn area(&self) -> f64 {
                PI * self.radius * self.radius
            }
            
            pub fn circumference(&self) -> f64 {
                2.0 * PI * self.radius
            }
        }
    }
    
    pub mod rectangle {
        pub struct Rectangle {
            width: f64,
            height: f64,
        }
        
        impl Rectangle {
            pub fn new(width: f64, height: f64) -> Rectangle {
                Rectangle { width, height }
            }
            
            pub fn area(&self) -> f64 {
                self.width * self.height
            }
            
            pub fn perimeter(&self) -> f64 {
                2.0 * (self.width + self.height)
            }
        }
    }
}

// Re-export commonly used types
pub use geometry::circle::Circle;
pub use geometry::rectangle::Rectangle;
```

### **Exercise 3: Using External Crates**

```rust
// Cargo.toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

// src/lib.rs
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct User {
    pub id: u32,
    pub name: String,
    pub email: String,
}

impl User {
    pub fn new(id: u32, name: String, email: String) -> User {
        User { id, name, email }
    }
    
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
    
    pub fn from_json(json: &str) -> Result<User, serde_json::Error> {
        serde_json::from_str(json)
    }
}

// src/main.rs
use my_crate::User;

fn main() {
    let user = User::new(1, "Alice".to_string(), "alice@example.com".to_string());
    
    match user.to_json() {
        Ok(json) => println!("JSON: {}", json),
        Err(e) => println!("Error: {}", e),
    }
}
```

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circle_area() {
        let circle = geometry::circle::Circle::new(5.0);
        let expected_area = std::f64::consts::PI * 25.0;
        assert!((circle.area() - expected_area).abs() < 0.001);
    }

    #[test]
    fn test_rectangle_perimeter() {
        let rect = geometry::rectangle::Rectangle::new(10.0, 20.0);
        assert_eq!(rect.perimeter(), 60.0);
    }

    #[test]
    fn test_user_serialization() {
        let user = User::new(1, "Alice".to_string(), "alice@example.com".to_string());
        
        let json = user.to_json().unwrap();
        let deserialized_user = User::from_json(&json).unwrap();
        
        assert_eq!(user.id, deserialized_user.id);
        assert_eq!(user.name, deserialized_user.name);
        assert_eq!(user.email, deserialized_user.email);
    }
}
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Missing pub Keyword**

```rust
// ❌ Wrong - function is private
mod math {
    fn add(a: i32, b: i32) -> i32 {  // Missing pub
        a + b
    }
}

fn main() {
    math::add(1, 2); // Error: function is private
}

// ✅ Correct - make function public
mod math {
    pub fn add(a: i32, b: i32) -> i32 {
        a + b
    }
}
```

### **Common Mistake 2: Incorrect Module Paths**

```rust
// ❌ Wrong - incorrect path
mod outer {
    pub mod inner {
        pub fn function() {}
    }
}

fn main() {
    outer::function(); // Error: function is in inner module
}

// ✅ Correct - use full path
fn main() {
    outer::inner::function();
}
```

### **Common Mistake 3: Circular Dependencies**

```rust
// ❌ Wrong - circular dependency
// mod_a.rs
use crate::mod_b::B;
pub struct A { b: B }

// mod_b.rs  
use crate::mod_a::A;
pub struct B { a: A }

// ✅ Correct - use references or separate common types
// common.rs
pub struct CommonData { /* ... */ }

// mod_a.rs
use crate::common::CommonData;
pub struct A { data: CommonData }
```

---

## 📦 **Crate Management**

### **Cargo.toml Configuration**

```toml
[package]
name = "my-awesome-crate"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <your.email@example.com>"]
description = "A sample Rust crate"
license = "MIT"
repository = "https://github.com/yourusername/my-awesome-crate"
keywords = ["example", "tutorial"]
categories = ["development-tools"]

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"], optional = true }

[dev-dependencies]
tempfile = "3.0"

[features]
default = []
async = ["tokio"]

[[bin]]
name = "my-tool"
path = "src/bin/my_tool.rs"
```

### **Publishing to crates.io**

```bash
# Login to crates.io
cargo login

# Check package before publishing
cargo package
cargo publish --dry-run

# Publish package
cargo publish
```

---

## 🎯 **Best Practices**

### **Module Organization**

```rust
// ✅ Good - clear hierarchy
pub mod api {
    pub mod v1 {
        pub mod users;
        pub mod posts;
    }
    pub mod v2 {
        pub mod users;
        pub mod posts;
    }
}

pub mod database {
    pub mod connection;
    pub mod models;
    pub mod migrations;
}

pub mod utils {
    pub mod validation;
    pub mod formatting;
}
```

### **Re-exports**

```rust
// ✅ Good - re-export commonly used types
pub mod internal {
    pub struct InternalType;
    pub struct AnotherInternalType;
}

// Re-export for easier use
pub use internal::{InternalType, AnotherInternalType};

// Or re-export with different names
pub use internal::InternalType as PublicType;
```

### **Documentation**

```rust
/// A user in the system
/// 
/// # Examples
/// 
/// ```
/// use my_crate::User;
/// 
/// let user = User::new(1, "Alice".to_string());
/// println!("User: {}", user.name);
/// ```
pub struct User {
    /// The unique identifier for the user
    pub id: u32,
    /// The user's display name
    pub name: String,
}
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [The Rust Book - Packages and Crates](https://doc.rust-lang.org/book/ch07-00-managing-growing-projects-with-packages-crates-and-modules.html) - Fetched: 2024-12-19T00:00:00Z
- [The Rust Book - Defining Modules](https://doc.rust-lang.org/book/ch07-02-defining-modules-to-control-scope-and-privacy.html) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Rust by Example - Modules](https://doc.rust-lang.org/rust-by-example/mod.html) - Fetched: 2024-12-19T00:00:00Z
- [Cargo Book](https://doc.rust-lang.org/cargo/) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. What's the difference between `pub` and private visibility?
2. How do you create a library crate vs a binary crate?
3. When should you use `use` statements vs full paths?
4. How do you organize modules in a large project?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Testing strategies and organization
- Documentation and examples
- Workspace management
- Advanced module patterns

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [7.2 Testing and Documentation](07_02_testing_documentation.md)
