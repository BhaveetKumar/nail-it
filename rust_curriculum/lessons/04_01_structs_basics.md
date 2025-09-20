# Lesson 4.1: Structs Basics

> **Module**: 4 - Structs, Enums & Pattern Matching  
> **Lesson**: 1 of 8  
> **Duration**: 2-3 hours  
> **Prerequisites**: Module 3 (Ownership and Borrowing)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Define and instantiate structs
- Understand different types of structs
- Implement methods and associated functions
- Use structs to organize related data
- Apply ownership principles to structs

---

## 🎯 **Overview**

Structs are custom data types that let you group related data together. They're similar to classes in other languages but without inheritance. Rust has three types of structs: regular structs, tuple structs, and unit structs.

---

## 🏗️ **Defining Structs**

### **Regular Structs**

```rust
struct User {
    username: String,
    email: String,
    age: u32,
    active: bool,
}

fn main() {
    let user1 = User {
        username: String::from("alice"),
        email: String::from("alice@example.com"),
        age: 30,
        active: true,
    };
    
    println!("User: {} ({})", user1.username, user1.email);
}
```

### **Tuple Structs**

```rust
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

fn main() {
    let black = Color(0, 0, 0);
    let origin = Point(0, 0, 0);
    
    println!("Black: RGB({}, {}, {})", black.0, black.1, black.2);
    println!("Origin: ({}, {}, {})", origin.0, origin.1, origin.2);
}
```

### **Unit Structs**

```rust
struct AlwaysEqual;

fn main() {
    let subject = AlwaysEqual;
    // Unit structs are useful for implementing traits
}
```

---

## 🔧 **Instantiating Structs**

### **Basic Instantiation**

```rust
struct Rectangle {
    width: u32,
    height: u32,
}

fn main() {
    let rect1 = Rectangle {
        width: 30,
        height: 50,
    };
    
    println!("Rectangle: {}x{}", rect1.width, rect1.height);
}
```

### **Field Init Shorthand**

```rust
fn build_user(username: String, email: String) -> User {
    User {
        username,    // shorthand for username: username
        email,       // shorthand for email: email
        age: 0,
        active: true,
    }
}

fn main() {
    let user = build_user(
        String::from("bob"),
        String::from("bob@example.com")
    );
    
    println!("User: {}", user.username);
}
```

### **Struct Update Syntax**

```rust
fn main() {
    let user1 = User {
        username: String::from("alice"),
        email: String::from("alice@example.com"),
        age: 30,
        active: true,
    };
    
    let user2 = User {
        username: String::from("bob"),
        email: String::from("bob@example.com"),
        ..user1  // Use remaining fields from user1
    };
    
    println!("User2: {} (age: {})", user2.username, user2.age);
}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Basic Struct**

```rust
struct Book {
    title: String,
    author: String,
    pages: u32,
    is_published: bool,
}

fn main() {
    let book = Book {
        title: String::from("The Rust Programming Language"),
        author: String::from("Steve Klabnik and Carol Nichols"),
        pages: 552,
        is_published: true,
    };
    
    println!("Book: {} by {}", book.title, book.author);
    println!("Pages: {}, Published: {}", book.pages, book.is_published);
}
```

### **Exercise 2: Tuple Struct**

```rust
struct RGB(u8, u8, u8);

impl RGB {
    fn new(r: u8, g: u8, b: u8) -> RGB {
        RGB(r, g, b)
    }
    
    fn to_hex(&self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.0, self.1, self.2)
    }
}

fn main() {
    let red = RGB::new(255, 0, 0);
    let green = RGB::new(0, 255, 0);
    let blue = RGB::new(0, 0, 255);
    
    println!("Red: {}", red.to_hex());
    println!("Green: {}", green.to_hex());
    println!("Blue: {}", blue.to_hex());
}
```

### **Exercise 3: Struct with Methods**

```rust
struct Circle {
    radius: f64,
}

impl Circle {
    fn new(radius: f64) -> Circle {
        Circle { radius }
    }
    
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
    
    fn circumference(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.radius
    }
}

fn main() {
    let circle = Circle::new(5.0);
    println!("Area: {:.2}", circle.area());
    println!("Circumference: {:.2}", circle.circumference());
}
```

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_creation() {
        let user = User {
            username: String::from("test"),
            email: String::from("test@example.com"),
            age: 25,
            active: true,
        };
        
        assert_eq!(user.username, "test");
        assert_eq!(user.email, "test@example.com");
        assert_eq!(user.age, 25);
        assert!(user.active);
    }

    #[test]
    fn test_rectangle_area() {
        let rect = Rectangle { width: 10, height: 20 };
        assert_eq!(rect.width, 10);
        assert_eq!(rect.height, 20);
    }

    #[test]
    fn test_circle_area() {
        let circle = Circle::new(5.0);
        let expected_area = std::f64::consts::PI * 25.0;
        assert!((circle.area() - expected_area).abs() < 0.001);
    }

    #[test]
    fn test_rgb_hex() {
        let red = RGB::new(255, 0, 0);
        assert_eq!(red.to_hex(), "#FF0000");
    }
}
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Forgetting String Ownership**

```rust
// ❌ Wrong - trying to use &str in struct
struct BadUser {
    username: &str,  // Error: missing lifetime parameter
    email: &str,
}

// ✅ Correct - use String for owned data
struct GoodUser {
    username: String,
    email: String,
}
```

### **Common Mistake 2: Moving Out of Struct**

```rust
// ❌ Wrong - moving out of struct
fn bad_example() {
    let user = User {
        username: String::from("alice"),
        email: String::from("alice@example.com"),
        age: 30,
        active: true,
    };
    
    let username = user.username; // Moves username out
    println!("{}", user.email);   // Error: user partially moved
}

// ✅ Correct - use references or clone
fn good_example() {
    let user = User {
        username: String::from("alice"),
        email: String::from("alice@example.com"),
        age: 30,
        active: true,
    };
    
    let username = &user.username; // Borrow instead of move
    println!("{}", user.email);    // This works now
}
```

### **Common Mistake 3: Confusing Methods and Functions**

```rust
impl Rectangle {
    // Associated function (like static method)
    fn new(width: u32, height: u32) -> Rectangle {
        Rectangle { width, height }
    }
    
    // Method (takes &self)
    fn area(&self) -> u32 {
        self.width * self.height
    }
}

fn main() {
    let rect = Rectangle::new(10, 20); // Associated function
    let area = rect.area();            // Method
    println!("Area: {}", area);
}
```

---

## 📊 **Advanced Struct Patterns**

### **Builder Pattern**

```rust
struct UserBuilder {
    username: Option<String>,
    email: Option<String>,
    age: Option<u32>,
    active: Option<bool>,
}

impl UserBuilder {
    fn new() -> UserBuilder {
        UserBuilder {
            username: None,
            email: None,
            age: None,
            active: None,
        }
    }
    
    fn username(mut self, username: String) -> UserBuilder {
        self.username = Some(username);
        self
    }
    
    fn email(mut self, email: String) -> UserBuilder {
        self.email = Some(email);
        self
    }
    
    fn age(mut self, age: u32) -> UserBuilder {
        self.age = Some(age);
        self
    }
    
    fn build(self) -> Result<User, String> {
        Ok(User {
            username: self.username.ok_or("Username required")?,
            email: self.email.ok_or("Email required")?,
            age: self.age.unwrap_or(0),
            active: self.active.unwrap_or(true),
        })
    }
}

fn main() {
    let user = UserBuilder::new()
        .username(String::from("alice"))
        .email(String::from("alice@example.com"))
        .age(30)
        .build()
        .unwrap();
    
    println!("User: {}", user.username);
}
```

---

## 🎯 **Best Practices**

### **Struct Design**

```rust
// ✅ Good - clear, descriptive names
struct UserAccount {
    user_id: u64,
    username: String,
    email_address: String,
    account_balance: f64,
    is_verified: bool,
}

// ✅ Good - group related fields
struct Address {
    street: String,
    city: String,
    state: String,
    zip_code: String,
    country: String,
}
```

### **Method Organization**

```rust
impl User {
    // Associated functions (constructors)
    fn new(username: String, email: String) -> User {
        User {
            username,
            email,
            age: 0,
            active: true,
        }
    }
    
    // Getters
    fn username(&self) -> &str {
        &self.username
    }
    
    fn email(&self) -> &str {
        &self.email
    }
    
    // Setters
    fn set_age(&mut self, age: u32) {
        self.age = age;
    }
    
    fn set_active(&mut self, active: bool) {
        self.active = active;
    }
    
    // Business logic methods
    fn is_adult(&self) -> bool {
        self.age >= 18
    }
}
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [The Rust Book - Structs](https://doc.rust-lang.org/book/ch05-00-structs.html) - Fetched: 2024-12-19T00:00:00Z
- [The Rust Book - Method Syntax](https://doc.rust-lang.org/book/ch05-03-method-syntax.html) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Rust by Example - Structs](https://doc.rust-lang.org/rust-by-example/custom_types/structs.html) - Fetched: 2024-12-19T00:00:00Z
- [Rustlings - Structs](https://github.com/rust-lang/rustlings) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. What are the three types of structs in Rust?
2. What's the difference between methods and associated functions?
3. When should you use `&self` vs `self` in method parameters?
4. How do you handle ownership when working with structs?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Enums and their variants
- Pattern matching with match expressions
- Option and Result enums
- Advanced enum patterns

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [4.2 Enums and Pattern Matching](04_02_enums_pattern_matching.md)
