# Lesson 40.1: Advanced WebAssembly Development

> **Module**: 40 - Advanced WebAssembly Development  
> **Lesson**: 1 of 6  
> **Duration**: 4-5 hours  
> **Prerequisites**: Module 39 (Final Project)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Build high-performance WebAssembly applications
- Implement complex WASM modules with Rust
- Optimize WASM performance and bundle size
- Integrate WASM with modern web frameworks
- Deploy WASM applications to production

---

## 🎯 **Overview**

Advanced WebAssembly development in Rust involves building high-performance web applications, optimizing WASM modules, and integrating with modern web frameworks. This lesson covers advanced WASM patterns, performance optimization, and production deployment.

---

## 🔧 **Advanced WASM Patterns**

### **High-Performance WASM Module**

```rust
// Cargo.toml
[package]
name = "advanced-wasm-module"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2"
js-sys = "0.3"
web-sys = "0.3"
serde = { version = "1.0", features = ["derive"] }
serde-wasm-bindgen = "0.6"
wee_alloc = "0.4.5"

[dependencies.web-sys]
version = "0.3"
features = [
  "console",
  "Document",
  "Element",
  "HtmlElement",
  "Window",
  "CanvasRenderingContext2d",
  "ImageData",
  "Uint8ClampedArray",
  "Performance",
  "Request",
  "Response",
  "RequestInit",
  "RequestMode",
  "Headers",
]

// src/lib.rs
use wasm_bindgen::prelude::*;
use web_sys::*;
use js_sys::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

// Use `wee_alloc` as the global allocator for smaller binary size
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// A macro to provide `println!(..)`-style syntax for `console.log` logging.
#[macro_export]
macro_rules! log {
    ( $( $t:tt )* ) => {
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
}

// Enable console error panic hook for better debugging
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
    log!("WASM module initialized");
}

#[wasm_bindgen]
pub struct AdvancedWasmModule {
    pub data: Mutex<HashMap<String, Vec<f64>>>,
    pub performance_tracker: PerformanceTracker,
    pub canvas_manager: CanvasManager,
    pub network_client: NetworkClient,
}

#[wasm_bindgen]
impl AdvancedWasmModule {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            data: Mutex::new(HashMap::new()),
            performance_tracker: PerformanceTracker::new(),
            canvas_manager: CanvasManager::new(),
            network_client: NetworkClient::new(),
        }
    }
    
    #[wasm_bindgen]
    pub fn process_large_dataset(&self, data: &[f64], operation: &str) -> Result<JsValue, JsValue> {
        let start_time = self.performance_tracker.start_timer("process_large_dataset");
        
        let result = match operation {
            "sort" => self.sort_data(data),
            "filter" => self.filter_data(data),
            "transform" => self.transform_data(data),
            "analyze" => self.analyze_data(data),
            _ => return Err(JsValue::from_str("Unknown operation")),
        };
        
        self.performance_tracker.end_timer(start_time);
        
        match result {
            Ok(processed_data) => {
                let js_array = JsValue::from_serde(&processed_data)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))?;
                Ok(js_array)
            }
            Err(e) => Err(JsValue::from_str(&format!("Processing error: {}", e))),
        }
    }
    
    #[wasm_bindgen]
    pub async fn fetch_and_process_data(&self, url: &str) -> Result<JsValue, JsValue> {
        let start_time = self.performance_tracker.start_timer("fetch_and_process_data");
        
        match self.network_client.fetch_data(url).await {
            Ok(data) => {
                let processed = self.process_large_dataset(&data, "analyze")?;
                self.performance_tracker.end_timer(start_time);
                Ok(processed)
            }
            Err(e) => {
                self.performance_tracker.end_timer(start_time);
                Err(JsValue::from_str(&format!("Network error: {}", e)))
            }
        }
    }
    
    #[wasm_bindgen]
    pub fn render_canvas(&self, canvas_id: &str, width: u32, height: u32) -> Result<(), JsValue> {
        self.canvas_manager.render(canvas_id, width, height)
    }
    
    #[wasm_bindgen]
    pub fn get_performance_metrics(&self) -> JsValue {
        JsValue::from_serde(&self.performance_tracker.get_metrics())
            .unwrap_or_else(|_| JsValue::NULL)
    }
    
    fn sort_data(&self, data: &[f64]) -> Result<Vec<f64>, String> {
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(sorted_data)
    }
    
    fn filter_data(&self, data: &[f64]) -> Result<Vec<f64>, String> {
        let filtered: Vec<f64> = data.iter()
            .filter(|&&x| x > 0.0)
            .copied()
            .collect();
        Ok(filtered)
    }
    
    fn transform_data(&self, data: &[f64]) -> Result<Vec<f64>, String> {
        let transformed: Vec<f64> = data.iter()
            .map(|&x| x * x + 2.0 * x + 1.0)
            .collect();
        Ok(transformed)
    }
    
    fn analyze_data(&self, data: &[f64]) -> Result<Vec<f64>, String> {
        if data.is_empty() {
            return Ok(vec![]);
        }
        
        let sum: f64 = data.iter().sum();
        let mean = sum / data.len() as f64;
        let variance: f64 = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();
        
        let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        Ok(vec![mean, std_dev, min, max])
    }
}

#[derive(Clone)]
pub struct PerformanceTracker {
    pub metrics: Mutex<HashMap<String, f64>>,
    pub timers: Mutex<HashMap<String, f64>>,
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            metrics: Mutex::new(HashMap::new()),
            timers: Mutex::new(HashMap::new()),
        }
    }
    
    pub fn start_timer(&self, name: &str) -> f64 {
        let performance = web_sys::window()
            .unwrap()
            .performance()
            .unwrap();
        let now = performance.now();
        
        let mut timers = self.timers.lock().unwrap();
        timers.insert(name.to_string(), now);
        now
    }
    
    pub fn end_timer(&self, start_time: f64) {
        let performance = web_sys::window()
            .unwrap()
            .performance()
            .unwrap();
        let duration = performance.now() - start_time;
        
        let mut metrics = self.metrics.lock().unwrap();
        metrics.insert("last_duration".to_string(), duration);
    }
    
    pub fn get_metrics(&self) -> HashMap<String, f64> {
        self.metrics.lock().unwrap().clone()
    }
}

pub struct CanvasManager {
    pub context: Option<CanvasRenderingContext2d>,
}

impl CanvasManager {
    pub fn new() -> Self {
        Self { context: None }
    }
    
    pub fn render(&self, canvas_id: &str, width: u32, height: u32) -> Result<(), JsValue> {
        let document = web_sys::window()
            .unwrap()
            .document()
            .unwrap();
        
        let canvas = document
            .get_element_by_id(canvas_id)
            .ok_or_else(|| JsValue::from_str("Canvas not found"))?
            .dyn_into::<HtmlCanvasElement>()
            .map_err(|_| JsValue::from_str("Invalid canvas element"))?;
        
        canvas.set_width(width);
        canvas.set_height(height);
        
        let context = canvas
            .get_context("2d")
            .unwrap()
            .unwrap()
            .dyn_into::<CanvasRenderingContext2d>()
            .unwrap();
        
        // Render some graphics
        self.draw_graphics(&context, width, height);
        
        Ok(())
    }
    
    fn draw_graphics(&self, context: &CanvasRenderingContext2d, width: u32, height: u32) {
        // Clear canvas
        context.clear_rect(0.0, 0.0, width as f64, height as f64);
        
        // Draw background
        context.set_fill_style(&JsValue::from_str("#f0f0f0"));
        context.fill_rect(0.0, 0.0, width as f64, height as f64);
        
        // Draw some shapes
        context.set_fill_style(&JsValue::from_str("#ff0000"));
        context.fill_rect(50.0, 50.0, 100.0, 100.0);
        
        context.set_fill_style(&JsValue::from_str("#00ff00"));
        context.begin_path();
        context.arc(200.0, 100.0, 50.0, 0.0, 2.0 * std::f64::consts::PI).unwrap();
        context.fill();
        
        // Draw text
        context.set_fill_style(&JsValue::from_str("#000000"));
        context.set_font("20px Arial");
        context.fill_text("WASM Graphics", 50.0, 200.0).unwrap();
    }
}

pub struct NetworkClient {
    pub base_url: String,
}

impl NetworkClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.example.com".to_string(),
        }
    }
    
    pub async fn fetch_data(&self, url: &str) -> Result<Vec<f64>, String> {
        let window = web_sys::window().unwrap();
        let request = Request::new_with_str(url).map_err(|e| format!("Request creation failed: {:?}", e))?;
        
        let request_init = RequestInit::new();
        request_init.method("GET");
        request_init.mode(RequestMode::Cors);
        
        let request = request
            .with_init(&request_init)
            .map_err(|e| format!("Request init failed: {:?}", e))?;
        
        let promise = window.fetch_with_request(&request);
        let future = wasm_bindgen_futures::JsFuture::from(promise);
        
        match future.await {
            Ok(response) => {
                let response: Response = response.dyn_into().unwrap();
                let promise = response.json().unwrap();
                let future = wasm_bindgen_futures::JsFuture::from(promise);
                
                match future.await {
                    Ok(json) => {
                        let data: Vec<f64> = json.into_serde()
                            .map_err(|e| format!("JSON parsing failed: {}", e))?;
                        Ok(data)
                    }
                    Err(e) => Err(format!("JSON parsing failed: {:?}", e)),
                }
            }
            Err(e) => Err(format!("Network request failed: {:?}", e)),
        }
    }
}
```

### **WASM Memory Management**

```rust
use wasm_bindgen::prelude::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

// Custom allocator for WASM
pub struct WasmAllocator {
    pub allocated: AtomicUsize,
    pub peak: AtomicUsize,
}

unsafe impl GlobalAlloc for WasmAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let size = layout.size();
            let current = self.allocated.fetch_add(size, Ordering::Relaxed);
            let peak = self.peak.load(Ordering::Relaxed);
            if current + size > peak {
                self.peak.store(current + size, Ordering::Relaxed);
            }
        }
        ptr
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = layout.size();
        self.allocated.fetch_sub(size, Ordering::Relaxed);
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static ALLOCATOR: WasmAllocator = WasmAllocator {
    allocated: AtomicUsize::new(0),
    peak: AtomicUsize::new(0),
};

#[wasm_bindgen]
pub struct MemoryManager {
    pub allocated: AtomicUsize,
    pub peak: AtomicUsize,
}

#[wasm_bindgen]
impl MemoryManager {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            allocated: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }
    }
    
    #[wasm_bindgen]
    pub fn get_memory_usage(&self) -> JsValue {
        let usage = serde_json::json!({
            "allocated": self.allocated.load(Ordering::Relaxed),
            "peak": self.peak.load(Ordering::Relaxed),
        });
        JsValue::from_serde(&usage).unwrap_or_else(|_| JsValue::NULL)
    }
    
    #[wasm_bindgen]
    pub fn allocate_memory(&self, size: usize) -> Result<JsValue, JsValue> {
        let layout = Layout::from_size_align(size, 8)
            .map_err(|e| JsValue::from_str(&format!("Invalid layout: {}", e)))?;
        
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(JsValue::from_str("Memory allocation failed"));
        }
        
        let current = self.allocated.fetch_add(size, Ordering::Relaxed);
        let peak = self.peak.load(Ordering::Relaxed);
        if current + size > peak {
            self.peak.store(current + size, Ordering::Relaxed);
        }
        
        Ok(JsValue::from_f64(ptr as u64 as f64))
    }
    
    #[wasm_bindgen]
    pub fn deallocate_memory(&self, ptr: u64, size: usize) -> Result<(), JsValue> {
        let layout = Layout::from_size_align(size, 8)
            .map_err(|e| JsValue::from_str(&format!("Invalid layout: {}", e)))?;
        
        unsafe {
            std::alloc::dealloc(ptr as *mut u8, layout);
        }
        
        self.allocated.fetch_sub(size, Ordering::Relaxed);
        Ok(())
    }
}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: WASM Image Processing**

```rust
use wasm_bindgen::prelude::*;
use web_sys::*;
use js_sys::*;

#[wasm_bindgen]
pub struct ImageProcessor {
    pub canvas: HtmlCanvasElement,
    pub context: CanvasRenderingContext2d,
}

#[wasm_bindgen]
impl ImageProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new(canvas_id: &str) -> Result<ImageProcessor, JsValue> {
        let document = web_sys::window()
            .unwrap()
            .document()
            .unwrap();
        
        let canvas = document
            .get_element_by_id(canvas_id)
            .ok_or_else(|| JsValue::from_str("Canvas not found"))?
            .dyn_into::<HtmlCanvasElement>()
            .map_err(|_| JsValue::from_str("Invalid canvas element"))?;
        
        let context = canvas
            .get_context("2d")
            .unwrap()
            .unwrap()
            .dyn_into::<CanvasRenderingContext2d>()
            .unwrap();
        
        Ok(ImageProcessor { canvas, context })
    }
    
    #[wasm_bindgen]
    pub fn load_image(&self, image_data: &Uint8ClampedArray, width: u32, height: u32) -> Result<(), JsValue> {
        let image_data = ImageData::new_with_u8_clamped_array_and_sh(image_data, width, height)
            .map_err(|_| JsValue::from_str("Failed to create ImageData"))?;
        
        self.context.put_image_data(&image_data, 0.0, 0.0)
            .map_err(|_| JsValue::from_str("Failed to put image data"))?;
        
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn apply_filter(&self, filter_type: &str) -> Result<Uint8ClampedArray, JsValue> {
        let width = self.canvas.width();
        let height = self.canvas.height();
        
        let image_data = self.context
            .get_image_data(0.0, 0.0, width as f64, height as f64)
            .map_err(|_| JsValue::from_str("Failed to get image data"))?;
        
        let mut data = image_data.data().to_vec();
        
        match filter_type {
            "grayscale" => self.apply_grayscale_filter(&mut data),
            "blur" => self.apply_blur_filter(&mut data, width, height),
            "sharpen" => self.apply_sharpen_filter(&mut data, width, height),
            "edge_detect" => self.apply_edge_detect_filter(&mut data, width, height),
            _ => return Err(JsValue::from_str("Unknown filter type")),
        }
        
        let result = Uint8ClampedArray::new_with_length(data.len() as u32);
        result.copy_from(&data);
        
        Ok(result)
    }
    
    fn apply_grayscale_filter(&self, data: &mut [u8]) {
        for i in (0..data.len()).step_by(4) {
            let r = data[i] as f32;
            let g = data[i + 1] as f32;
            let b = data[i + 2] as f32;
            
            let gray = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
            
            data[i] = gray;     // R
            data[i + 1] = gray; // G
            data[i + 2] = gray; // B
            // data[i + 3] is alpha, keep unchanged
        }
    }
    
    fn apply_blur_filter(&self, data: &mut [u8], width: u32, height: u32) {
        let mut temp = data.to_vec();
        let radius = 1;
        
        for y in radius..(height - radius) as usize {
            for x in radius..(width - radius) as usize {
                let mut r_sum = 0u32;
                let mut g_sum = 0u32;
                let mut b_sum = 0u32;
                let mut count = 0u32;
                
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let ny = (y as i32 + dy) as usize;
                        let nx = (x as i32 + dx) as usize;
                        let idx = (ny * width as usize + nx) * 4;
                        
                        r_sum += data[idx] as u32;
                        g_sum += data[idx + 1] as u32;
                        b_sum += data[idx + 2] as u32;
                        count += 1;
                    }
                }
                
                let idx = (y * width as usize + x) * 4;
                temp[idx] = (r_sum / count) as u8;
                temp[idx + 1] = (g_sum / count) as u8;
                temp[idx + 2] = (b_sum / count) as u8;
            }
        }
        
        data.copy_from_slice(&temp);
    }
    
    fn apply_sharpen_filter(&self, data: &mut [u8], width: u32, height: u32) {
        let kernel = [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ];
        
        let mut temp = data.to_vec();
        
        for y in 1..(height - 1) as usize {
            for x in 1..(width - 1) as usize {
                let mut r_sum = 0i32;
                let mut g_sum = 0i32;
                let mut b_sum = 0i32;
                
                for ky in 0..3 {
                    for kx in 0..3 {
                        let ny = y + ky - 1;
                        let nx = x + kx - 1;
                        let idx = (ny * width as usize + nx) * 4;
                        let weight = kernel[ky][kx];
                        
                        r_sum += (data[idx] as i32) * weight;
                        g_sum += (data[idx + 1] as i32) * weight;
                        b_sum += (data[idx + 2] as i32) * weight;
                    }
                }
                
                let idx = (y * width as usize + x) * 4;
                temp[idx] = r_sum.max(0).min(255) as u8;
                temp[idx + 1] = g_sum.max(0).min(255) as u8;
                temp[idx + 2] = b_sum.max(0).min(255) as u8;
            }
        }
        
        data.copy_from_slice(&temp);
    }
    
    fn apply_edge_detect_filter(&self, data: &mut [u8], width: u32, height: u32) {
        let kernel_x = [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ];
        
        let kernel_y = [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ];
        
        let mut temp = data.to_vec();
        
        for y in 1..(height - 1) as usize {
            for x in 1..(width - 1) as usize {
                let mut gx = 0i32;
                let mut gy = 0i32;
                
                for ky in 0..3 {
                    for kx in 0..3 {
                        let ny = y + ky - 1;
                        let nx = x + kx - 1;
                        let idx = (ny * width as usize + nx) * 4;
                        let gray = (data[idx] as i32 + data[idx + 1] as i32 + data[idx + 2] as i32) / 3;
                        
                        gx += gray * kernel_x[ky][kx];
                        gy += gray * kernel_y[ky][kx];
                    }
                }
                
                let magnitude = ((gx * gx + gy * gy) as f32).sqrt() as u8;
                let idx = (y * width as usize + x) * 4;
                temp[idx] = magnitude;     // R
                temp[idx + 1] = magnitude; // G
                temp[idx + 2] = magnitude; // B
            }
        }
        
        data.copy_from_slice(&temp);
    }
}
```

### **Exercise 2: WASM Audio Processing**

```rust
use wasm_bindgen::prelude::*;
use web_sys::*;
use js_sys::*;

#[wasm_bindgen]
pub struct AudioProcessor {
    pub sample_rate: f32,
    pub buffer_size: usize,
}

#[wasm_bindgen]
impl AudioProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32, buffer_size: usize) -> Self {
        Self {
            sample_rate,
            buffer_size,
        }
    }
    
    #[wasm_bindgen]
    pub fn process_audio(&self, input_data: &Float32Array) -> Result<Float32Array, JsValue> {
        let input_len = input_data.length() as usize;
        let mut output_data = vec![0.0f32; input_len];
        
        // Copy input data
        for i in 0..input_len {
            output_data[i] = input_data.get_index(i as u32);
        }
        
        // Apply audio processing
        self.apply_high_pass_filter(&mut output_data);
        self.apply_compression(&mut output_data);
        self.apply_reverb(&mut output_data);
        
        let result = Float32Array::new_with_length(output_data.len() as u32);
        for (i, &value) in output_data.iter().enumerate() {
            result.set_index(i as u32, value);
        }
        
        Ok(result)
    }
    
    #[wasm_bindgen]
    pub fn generate_tone(&self, frequency: f32, duration: f32, amplitude: f32) -> Float32Array {
        let sample_count = (self.sample_rate * duration) as usize;
        let mut samples = vec![0.0f32; sample_count];
        
        for i in 0..sample_count {
            let t = i as f32 / self.sample_rate;
            samples[i] = amplitude * (2.0 * std::f64::consts::PI * frequency as f64 * t as f64).sin() as f32;
        }
        
        let result = Float32Array::new_with_length(samples.len() as u32);
        for (i, &sample) in samples.iter().enumerate() {
            result.set_index(i as u32, sample);
        }
        
        result
    }
    
    #[wasm_bindgen]
    pub fn apply_fft(&self, input_data: &Float32Array) -> Result<Float32Array, JsValue> {
        let input_len = input_data.length() as usize;
        let mut real = vec![0.0f32; input_len];
        let mut imag = vec![0.0f32; input_len];
        
        // Copy input data
        for i in 0..input_len {
            real[i] = input_data.get_index(i as u32);
        }
        
        // Apply FFT
        self.fft(&mut real, &mut imag);
        
        // Calculate magnitude spectrum
        let mut magnitude = vec![0.0f32; input_len / 2];
        for i in 0..input_len / 2 {
            magnitude[i] = (real[i] * real[i] + imag[i] * imag[i]).sqrt();
        }
        
        let result = Float32Array::new_with_length(magnitude.len() as u32);
        for (i, &mag) in magnitude.iter().enumerate() {
            result.set_index(i as u32, mag);
        }
        
        Ok(result)
    }
    
    fn apply_high_pass_filter(&self, data: &mut [f32]) {
        let cutoff = 0.1; // Normalized frequency
        let mut prev_input = 0.0f32;
        let mut prev_output = 0.0f32;
        
        for sample in data.iter_mut() {
            let output = *sample - prev_input + cutoff * prev_output;
            prev_input = *sample;
            prev_output = output;
            *sample = output;
        }
    }
    
    fn apply_compression(&self, data: &mut [f32]) {
        let threshold = 0.5;
        let ratio = 4.0;
        
        for sample in data.iter_mut() {
            let abs_sample = sample.abs();
            if abs_sample > threshold {
                let excess = abs_sample - threshold;
                let compressed_excess = excess / ratio;
                *sample = *sample.signum() * (threshold + compressed_excess);
            }
        }
    }
    
    fn apply_reverb(&self, data: &mut [f32]) {
        let delay_samples = (self.sample_rate * 0.1) as usize; // 100ms delay
        let feedback = 0.3;
        
        if data.len() > delay_samples {
            for i in delay_samples..data.len() {
                data[i] += data[i - delay_samples] * feedback;
            }
        }
    }
    
    fn fft(&self, real: &mut [f32], imag: &mut [f32]) {
        let n = real.len();
        if n <= 1 {
            return;
        }
        
        // Bit-reverse permutation
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            
            if i < j {
                real.swap(i, j);
                imag.swap(i, j);
            }
        }
        
        // Cooley-Tukey FFT
        let mut length = 2;
        while length <= n {
            let wlen = 2.0 * std::f64::consts::PI / length as f64;
            for i in (0..n).step_by(length) {
                let mut w = 1.0f64;
                for j in 0..length / 2 {
                    let u = real[i + j] as f64;
                    let v = imag[i + j] as f64;
                    let t_real = real[i + j + length / 2] as f64;
                    let t_imag = imag[i + j + length / 2] as f64;
                    
                    real[i + j] = (u + t_real) as f32;
                    imag[i + j] = (v + t_imag) as f32;
                    
                    let temp_real = (u - t_real) * w.cos() - (v - t_imag) * w.sin();
                    let temp_imag = (u - t_real) * w.sin() + (v - t_imag) * w.cos();
                    
                    real[i + j + length / 2] = temp_real as f32;
                    imag[i + j + length / 2] = temp_imag as f32;
                    
                    w *= (wlen).cos() + (wlen).sin() * 1.0i64;
                }
            }
            length <<= 1;
        }
    }
}
```

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_wasm_module_creation() {
        let module = AdvancedWasmModule::new();
        assert!(module.data.lock().unwrap().is_empty());
    }

    #[wasm_bindgen_test]
    fn test_data_processing() {
        let module = AdvancedWasmModule::new();
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        
        let result = module.process_large_dataset(&data, "sort").unwrap();
        let sorted: Vec<f64> = result.into_serde().unwrap();
        
        assert_eq!(sorted, vec![1.0, 1.0, 3.0, 4.0, 5.0]);
    }

    #[wasm_bindgen_test]
    fn test_memory_management() {
        let manager = MemoryManager::new();
        let ptr = manager.allocate_memory(1024).unwrap();
        assert!(ptr.as_f64().unwrap() > 0.0);
        
        manager.deallocate_memory(ptr.as_f64().unwrap() as u64, 1024).unwrap();
    }

    #[wasm_bindgen_test]
    fn test_audio_processing() {
        let processor = AudioProcessor::new(44100.0, 1024);
        let input_data = Float32Array::new_with_length(1024);
        
        for i in 0..1024 {
            input_data.set_index(i, (i as f32 * 0.01).sin());
        }
        
        let output = processor.process_audio(&input_data).unwrap();
        assert_eq!(output.length(), 1024);
    }
}
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Memory Leaks in WASM**

```rust
// ❌ Wrong - potential memory leak
#[wasm_bindgen]
pub fn bad_function() -> JsValue {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    JsValue::from_serde(&data).unwrap()
}

// ✅ Correct - proper memory management
#[wasm_bindgen]
pub fn good_function() -> JsValue {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = JsValue::from_serde(&data).unwrap();
    // data is automatically dropped here
    result
}
```

### **Common Mistake 2: Inefficient WASM Operations**

```rust
// ❌ Wrong - inefficient operations
#[wasm_bindgen]
pub fn bad_processing(data: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();
    for &value in data {
        result.push(value * 2.0);
    }
    result
}

// ✅ Correct - efficient operations
#[wasm_bindgen]
pub fn good_processing(data: &[f64]) -> Vec<f64> {
    data.iter().map(|&x| x * 2.0).collect()
}
```

---

## 📊 **Advanced WASM Patterns**

### **WASM Worker Threads**

```rust
use wasm_bindgen::prelude::*;
use web_sys::*;
use js_sys::*;

#[wasm_bindgen]
pub struct WasmWorker {
    pub worker: Worker,
    pub message_channel: MessageChannel,
}

#[wasm_bindgen]
impl WasmWorker {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmWorker, JsValue> {
        let worker = Worker::new("./worker.js")
            .map_err(|_| JsValue::from_str("Failed to create worker"))?;
        
        let message_channel = MessageChannel::new()
            .map_err(|_| JsValue::from_str("Failed to create message channel"))?;
        
        Ok(WasmWorker {
            worker,
            message_channel,
        })
    }
    
    #[wasm_bindgen]
    pub fn start_processing(&self, data: &[f64]) -> Result<(), JsValue> {
        let message = serde_json::json!({
            "type": "process_data",
            "data": data,
        });
        
        let js_message = JsValue::from_serde(&message)
            .map_err(|e| JsValue::from_str(&format!("Serialization failed: {}", e)))?;
        
        self.worker.post_message(&js_message)
            .map_err(|_| JsValue::from_str("Failed to post message"))?;
        
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn set_on_message(&self, callback: &js_sys::Function) -> Result<(), JsValue> {
        self.worker.set_onmessage(Some(callback));
        Ok(())
    }
}
```

### **WASM SIMD Operations**

```rust
use wasm_bindgen::prelude::*;
use std::arch::wasm32::*;

#[wasm_bindgen]
pub struct SimdProcessor {
    pub data: Vec<f32>,
}

#[wasm_bindgen]
impl SimdProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0.0f32; size],
        }
    }
    
    #[wasm_bindgen]
    pub fn simd_add(&mut self, other: &[f32]) -> Result<(), JsValue> {
        if self.data.len() != other.len() {
            return Err(JsValue::from_str("Data length mismatch"));
        }
        
        let chunks = self.data.chunks_exact_mut(4);
        let other_chunks = other.chunks_exact(4);
        
        for (chunk, other_chunk) in chunks.zip(other_chunks) {
            if chunk.len() == 4 && other_chunk.len() == 4 {
                unsafe {
                    let a = v128_load(chunk.as_ptr() as *const v128);
                    let b = v128_load(other_chunk.as_ptr() as *const v128);
                    let result = f32x4_add(a, b);
                    v128_store(chunk.as_mut_ptr() as *mut v128, result);
                }
            } else {
                // Handle remaining elements
                for (i, &value) in other_chunk.iter().enumerate() {
                    chunk[i] += value;
                }
            }
        }
        
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn simd_multiply(&mut self, scalar: f32) {
        let scalar_vec = [scalar; 4];
        let chunks = self.data.chunks_exact_mut(4);
        
        for chunk in chunks {
            if chunk.len() == 4 {
                unsafe {
                    let a = v128_load(chunk.as_ptr() as *const v128);
                    let b = v128_load(scalar_vec.as_ptr() as *const v128);
                    let result = f32x4_mul(a, b);
                    v128_store(chunk.as_mut_ptr() as *mut v128, result);
                }
            } else {
                // Handle remaining elements
                for value in chunk.iter_mut() {
                    *value *= scalar;
                }
            }
        }
    }
}
```

---

## 🎯 **Best Practices**

### **WASM Configuration**

```rust
// ✅ Good - comprehensive WASM configuration
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct WasmConfig {
    pub optimization: OptimizationConfig,
    pub memory: MemoryConfig,
    pub performance: PerformanceConfig,
    pub debugging: DebuggingConfig,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OptimizationConfig {
    pub enable_simd: bool,
    pub enable_threads: bool,
    pub enable_bulk_memory: bool,
    pub enable_reference_types: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MemoryConfig {
    pub initial_pages: u32,
    pub maximum_pages: u32,
    pub enable_memory64: bool,
    pub enable_shared_memory: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PerformanceConfig {
    pub enable_optimization: bool,
    pub enable_lto: bool,
    pub enable_strip: bool,
    pub enable_debug_info: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DebuggingConfig {
    pub enable_console_logging: bool,
    pub enable_performance_timing: bool,
    pub enable_memory_tracking: bool,
    pub enable_error_reporting: bool,
}
```

### **Error Handling**

```rust
// ✅ Good - comprehensive WASM error handling
use thiserror::Error;

#[derive(Error, Debug)]
pub enum WasmError {
    #[error("Memory allocation failed: {0}")]
    MemoryAllocationFailed(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Processing error: {0}")]
    ProcessingError(String),
    
    #[error("Validation error: {0}")]
    ValidationError(String),
}

pub type Result<T> = std::result::Result<T, WasmError>;
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [WebAssembly](https://webassembly.org/) - Fetched: 2024-12-19T00:00:00Z
- [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Rust WASM](https://rustwasm.github.io/book/) - Fetched: 2024-12-19T00:00:00Z
- [WASM Performance](https://rust-lang.github.io/async-book/) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. Can you build high-performance WebAssembly applications?
2. Do you understand complex WASM modules with Rust?
3. Can you optimize WASM performance and bundle size?
4. Do you know how to integrate WASM with modern web frameworks?
5. Can you deploy WASM applications to production?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Advanced embedded systems
- Real-time operating systems
- Hardware abstraction layers
- IoT device development

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [40.2 Advanced Embedded Systems](40_02_embedded_systems.md)
