# Lesson 20.1: WebAssembly (WASM) Basics

> **Module**: 20 - WebAssembly  
> **Lesson**: 1 of 6  
> **Duration**: 3-4 hours  
> **Prerequisites**: Module 19 (Game Development)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Understand WebAssembly and its benefits
- Set up Rust for WASM development
- Compile Rust code to WebAssembly
- Interact with JavaScript from Rust
- Build web applications with Rust and WASM

---

## 🎯 **Overview**

WebAssembly (WASM) is a binary instruction format that allows you to run high-performance code in web browsers. Rust has excellent support for WebAssembly, making it possible to write web applications entirely in Rust.

---

## 🔧 **Setting Up WASM Development**

### **Installation and Setup**

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Install cargo-generate
cargo install cargo-generate

# Create a new WASM project
cargo generate --git https://github.com/rustwasm/wasm-pack-template
```

### **Basic WASM Project Structure**

```
wasm-project/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   └── utils.rs
├── pkg/
│   ├── wasm_project.js
│   ├── wasm_project_bg.wasm
│   └── wasm_project.d.ts
└── www/
    ├── index.html
    ├── index.js
    └── style.css
```

---

## 🚀 **Basic WASM Functions**

### **Simple Math Functions**

```rust
// src/lib.rs
use wasm_bindgen::prelude::*;

// Import the `console.log` function
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

// Define a macro to make console.log easier to use
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

// Export a function to JavaScript
#[wasm_bindgen]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

// Export a function that takes a string
#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

// Export a function that uses console.log
#[wasm_bindgen]
pub fn log_hello() {
    console_log!("Hello from Rust!");
}

// Export a function that processes arrays
#[wasm_bindgen]
pub fn sum_array(numbers: &[i32]) -> i32 {
    numbers.iter().sum()
}
```

### **Cargo.toml Configuration**

```toml
[package]
name = "wasm-project"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2.87"
js-sys = "0.3.64"

[dependencies.web-sys]
version = "0.3.64"
features = [
  "console",
  "Document",
  "Element",
  "HtmlElement",
  "Window",
]
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Calculator Functions**

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Calculator {
    value: f64,
}

#[wasm_bindgen]
impl Calculator {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Calculator {
        Calculator { value: 0.0 }
    }
    
    #[wasm_bindgen]
    pub fn add(&mut self, n: f64) -> f64 {
        self.value += n;
        self.value
    }
    
    #[wasm_bindgen]
    pub fn subtract(&mut self, n: f64) -> f64 {
        self.value -= n;
        self.value
    }
    
    #[wasm_bindgen]
    pub fn multiply(&mut self, n: f64) -> f64 {
        self.value *= n;
        self.value
    }
    
    #[wasm_bindgen]
    pub fn divide(&mut self, n: f64) -> f64 {
        if n != 0.0 {
            self.value /= n;
        }
        self.value
    }
    
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.value = 0.0;
    }
    
    #[wasm_bindgen]
    pub fn get_value(&self) -> f64 {
        self.value
    }
}
```

### **Exercise 2: DOM Manipulation**

```rust
use wasm_bindgen::prelude::*;
use web_sys::{Document, Element, HtmlElement, Window};

#[wasm_bindgen]
pub fn create_button(text: &str) -> Result<Element, JsValue> {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    
    let button = document.create_element("button")?;
    button.set_text_content(Some(text));
    
    // Add click event listener
    let closure = Closure::wrap(Box::new(move || {
        web_sys::console::log_1(&"Button clicked!".into());
    }) as Box<dyn FnMut()>);
    
    button.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
    closure.forget();
    
    Ok(button)
}

#[wasm_bindgen]
pub fn append_to_body(element: &Element) -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let body = document.body().unwrap();
    
    body.append_child(element)?;
    Ok(())
}
```

### **Exercise 3: Canvas Drawing**

```rust
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

#[wasm_bindgen]
pub fn draw_circle(
    canvas: &HtmlCanvasElement,
    x: f64,
    y: f64,
    radius: f64,
    color: &str,
) -> Result<(), JsValue> {
    let context = canvas
        .get_context("2d")?
        .unwrap()
        .dyn_into::<CanvasRenderingContext2d>()?;
    
    context.begin_path();
    context.arc(x, y, radius, 0.0, 2.0 * std::f64::consts::PI)?;
    context.set_fill_style(&color.into());
    context.fill()?;
    
    Ok(())
}

#[wasm_bindgen]
pub fn draw_rectangle(
    canvas: &HtmlCanvasElement,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    color: &str,
) -> Result<(), JsValue> {
    let context = canvas
        .get_context("2d")?
        .unwrap()
        .dyn_into::<CanvasRenderingContext2d>()?;
    
    context.set_fill_style(&color.into());
    context.fill_rect(x, y, width, height);
    
    Ok(())
}
```

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculator_operations() {
        let mut calc = Calculator::new();
        
        assert_eq!(calc.add(5.0), 5.0);
        assert_eq!(calc.multiply(2.0), 10.0);
        assert_eq!(calc.subtract(3.0), 7.0);
        assert_eq!(calc.divide(2.0), 3.5);
    }

    #[test]
    fn test_calculator_clear() {
        let mut calc = Calculator::new();
        calc.add(10.0);
        calc.clear();
        assert_eq!(calc.get_value(), 0.0);
    }
}
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Missing wasm_bindgen Attribute**

```rust
// ❌ Wrong - function not exported to JavaScript
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

// ✅ Correct - use wasm_bindgen attribute
#[wasm_bindgen]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

### **Common Mistake 2: Incorrect Type Conversions**

```rust
// ❌ Wrong - String not supported
#[wasm_bindgen]
pub fn bad_function(s: String) -> String {
    s
}

// ✅ Correct - use &str and return String
#[wasm_bindgen]
pub fn good_function(s: &str) -> String {
    s.to_string()
}
```

### **Common Mistake 3: Forgetting to Handle Errors**

```rust
// ❌ Wrong - panic in WASM
#[wasm_bindgen]
pub fn bad_dom_operation() -> Element {
    let document = web_sys::window().unwrap().document().unwrap();
    document.create_element("div").unwrap()
}

// ✅ Correct - return Result
#[wasm_bindgen]
pub fn good_dom_operation() -> Result<Element, JsValue> {
    let window = web_sys::window().ok_or("No window")?;
    let document = window.document().ok_or("No document")?;
    document.create_element("div")
}
```

---

## 📊 **Advanced WASM Patterns**

### **Memory Management**

```rust
use wasm_bindgen::prelude::*;
use std::collections::HashMap;

#[wasm_bindgen]
pub struct DataStore {
    data: HashMap<String, String>,
}

#[wasm_bindgen]
impl DataStore {
    #[wasm_bindgen(constructor)]
    pub fn new() -> DataStore {
        DataStore {
            data: HashMap::new(),
        }
    }
    
    #[wasm_bindgen]
    pub fn set(&mut self, key: &str, value: &str) {
        self.data.insert(key.to_string(), value.to_string());
    }
    
    #[wasm_bindgen]
    pub fn get(&self, key: &str) -> Option<String> {
        self.data.get(key).cloned()
    }
    
    #[wasm_bindgen]
    pub fn keys(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }
}
```

### **Async Operations**

```rust
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

#[wasm_bindgen]
pub async fn fetch_data(url: &str) -> Result<String, JsValue> {
    let mut opts = RequestInit::new();
    opts.method("GET");
    opts.mode(RequestMode::Cors);
    
    let request = Request::new_with_str_and_init(url, &opts)?;
    let window = web_sys::window().unwrap();
    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
    let resp: Response = resp_value.dyn_into()?;
    
    let text = JsFuture::from(resp.text()?).await?;
    Ok(text.as_string().unwrap())
}
```

---

## 🎯 **Best Practices**

### **Performance Optimization**

```rust
// ✅ Good - use efficient data structures
use std::collections::HashMap;

#[wasm_bindgen]
pub struct EfficientCache {
    cache: HashMap<String, Vec<u8>>,
}

// ✅ Good - avoid unnecessary allocations
#[wasm_bindgen]
pub fn process_data(data: &[u8]) -> Vec<u8> {
    data.iter().map(|&x| x * 2).collect()
}

// ❌ Avoid - expensive operations in hot paths
#[wasm_bindgen]
pub fn bad_performance() -> String {
    let mut result = String::new();
    for i in 0..10000 {
        result.push_str(&format!("{}", i));
    }
    result
}
```

### **Error Handling**

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn safe_operation(input: &str) -> Result<String, JsValue> {
    if input.is_empty() {
        return Err("Input cannot be empty".into());
    }
    
    if input.len() > 1000 {
        return Err("Input too long".into());
    }
    
    Ok(input.to_uppercase())
}
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [Rust and WebAssembly](https://rustwasm.github.io/docs/book/) - Fetched: 2024-12-19T00:00:00Z
- [wasm-bindgen Guide](https://rustwasm.github.io/wasm-bindgen/) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [WebAssembly Examples](https://github.com/rustwasm/wasm-pack-template) - Fetched: 2024-12-19T00:00:00Z
- [Yew Framework](https://yew.rs/) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. What are the benefits of using WebAssembly in web applications?
2. How do you export Rust functions to JavaScript?
3. What are the key considerations for memory management in WASM?
4. How do you handle errors in WebAssembly applications?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Advanced WASM patterns and optimization
- Integration with modern web frameworks
- Performance profiling and debugging
- Building complete web applications

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [20.2 Advanced WASM Patterns](20_02_advanced_wasm.md)
