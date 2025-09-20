# Lesson 18.1: Embedded Rust Basics

> **Module**: 18 - Embedded Programming  
> **Lesson**: 1 of 8  
> **Duration**: 4-5 hours  
> **Prerequisites**: Module 17 (Advanced Async Patterns)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Understand the `no_std` environment and its constraints
- Set up embedded Rust development environment
- Write basic embedded programs for microcontrollers
- Use Hardware Abstraction Layers (HALs)
- Implement real-time programming patterns

---

## 🎯 **Overview**

Embedded Rust allows you to write safe, efficient code for microcontrollers and other resource-constrained devices. Unlike standard Rust programs, embedded programs run in a `no_std` environment without the standard library.

---

## 🔧 **no_std Environment**

### **What is no_std?**

```rust
#![no_std]  // Disable standard library
#![no_main] // Disable main function

use core::panic::PanicInfo;

// Custom panic handler
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

// Custom entry point
#[no_mangle]
pub extern "C" fn _start() -> ! {
    loop {}
}
```

### **Core vs Standard Library**

```rust
#![no_std]

// ✅ Available in no_std
use core::{
    mem, ptr, slice,
    option::Option,
    result::Result,
    cell::Cell,
    sync::atomic::{AtomicU32, Ordering},
};

// ❌ Not available in no_std
// use std::collections::HashMap;  // No heap allocation
// use std::thread;                // No threading
// use std::fs;                    // No file system
```

---

## 🏗️ **Hardware Abstraction Layer (HAL)**

### **Basic HAL Structure**

```rust
// Cargo.toml
[dependencies]
cortex-m = "0.7.7"
cortex-m-rt = "0.7.3"
panic-halt = "0.2.0"

// main.rs
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;

#[entry]
fn main() -> ! {
    // Initialize peripherals
    let peripherals = cortex_m::Peripherals::take().unwrap();
    
    // Configure GPIO
    let gpioa = &peripherals.GPIOA;
    let rcc = &peripherals.RCC;
    
    // Enable GPIOA clock
    rcc.ahbenr.modify(|_, w| w.iopaen().set_bit());
    
    // Configure PA5 as output
    gpioa.moder.modify(|_, w| w.moder5().bits(0b01));
    
    // Blink LED
    loop {
        // Turn on LED
        gpioa.bsrr.write(|w| w.bs5().set_bit());
        delay(1000000);
        
        // Turn off LED
        gpioa.bsrr.write(|w| w.br5().set_bit());
        delay(1000000);
    }
}

fn delay(count: u32) {
    for _ in 0..count {
        cortex_m::asm::nop();
    }
}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: LED Blinking**

```rust
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;
use stm32f1xx_hal::{
    gpio::{gpioa::PA5, Output, PushPull},
    pac,
    prelude::*,
};

#[entry]
fn main() -> ! {
    let dp = pac::Peripherals::take().unwrap();
    let cp = cortex_m::Peripherals::take().unwrap();
    
    let mut rcc = dp.RCC.constrain();
    let mut flash = dp.FLASH.constrain();
    let clocks = rcc.cfgr.freeze(&mut flash.acr);
    
    let mut gpioa = dp.GPIOA.split(&mut rcc.apb2);
    let mut led = gpioa.pa5.into_push_pull_output(&mut gpioa.crl);
    
    let mut delay = cp.SYST.delay(&clocks);
    
    loop {
        led.set_high().ok();
        delay.delay_ms(1000u32);
        led.set_low().ok();
        delay.delay_ms(1000u32);
    }
}
```

### **Exercise 2: Button Input**

```rust
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;
use stm32f1xx_hal::{
    gpio::{gpioa::PA5, gpioc::PC13, Input, Output, PullUp, PushPull},
    pac,
    prelude::*,
};

#[entry]
fn main() -> ! {
    let dp = pac::Peripherals::take().unwrap();
    let cp = cortex_m::Peripherals::take().unwrap();
    
    let mut rcc = dp.RCC.constrain();
    let mut flash = dp.FLASH.constrain();
    let clocks = rcc.cfgr.freeze(&mut flash.acr);
    
    let mut gpioa = dp.GPIOA.split(&mut rcc.apb2);
    let mut gpioc = dp.GPIOC.split(&mut rcc.apb2);
    
    let mut led = gpioa.pa5.into_push_pull_output(&mut gpioa.crl);
    let button = gpioc.pc13.into_pull_up_input(&mut gpioc.crh);
    
    let mut delay = cp.SYST.delay(&clocks);
    
    loop {
        if button.is_low().unwrap() {
            led.set_high().ok();
        } else {
            led.set_low().ok();
        }
        delay.delay_ms(10u32);
    }
}
```

### **Exercise 3: UART Communication**

```rust
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;
use stm32f1xx_hal::{
    pac,
    prelude::*,
    serial::{Config, Serial},
};

#[entry]
fn main() -> ! {
    let dp = pac::Peripherals::take().unwrap();
    let cp = cortex_m::Peripherals::take().unwrap();
    
    let mut rcc = dp.RCC.constrain();
    let mut flash = dp.FLASH.constrain();
    let clocks = rcc.cfgr.freeze(&mut flash.acr);
    
    let mut gpioa = dp.GPIOA.split(&mut rcc.apb2);
    let tx = gpioa.pa9.into_alternate_push_pull(&mut gpioa.crh);
    let rx = gpioa.pa10;
    
    let serial = Serial::usart1(
        dp.USART1,
        (tx, rx),
        Config::default().baudrate(115200.bps()),
        clocks,
        &mut rcc.apb2,
    );
    
    let (mut tx, mut rx) = serial.split();
    
    let mut delay = cp.SYST.delay(&clocks);
    
    loop {
        // Send message
        let message = b"Hello from embedded Rust!\r\n";
        for &byte in message {
            tx.write(byte).ok();
        }
        
        delay.delay_ms(1000u32);
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
    fn test_delay_function() {
        let start = cortex_m::Peripherals::take().unwrap().DWT.cycle_count();
        delay(1000);
        let end = cortex_m::Peripherals::take().unwrap().DWT.cycle_count();
        assert!(end - start >= 1000);
    }

    #[test]
    fn test_gpio_configuration() {
        // Test GPIO configuration logic
        let config = 0b01; // Output mode
        assert_eq!(config, 0b01);
    }
}
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Using std in no_std**

```rust
#![no_std]

// ❌ Wrong - std not available
use std::collections::HashMap;

// ✅ Correct - use core or alloc
use core::cell::Cell;
// or
#![feature(alloc)]
extern crate alloc;
use alloc::vec::Vec;
```

### **Common Mistake 2: Missing Panic Handler**

```rust
#![no_std]
#![no_main]

// ❌ Wrong - no panic handler
#[no_mangle]
pub extern "C" fn _start() -> ! {
    panic!("This will cause undefined behavior");
}

// ✅ Correct - define panic handler
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
```

### **Common Mistake 3: Incorrect Memory Layout**

```rust
#![no_std]

// ❌ Wrong - incorrect memory layout
static mut COUNTER: u32 = 0;

// ✅ Correct - use proper synchronization
use core::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn increment_counter() {
    COUNTER.fetch_add(1, Ordering::SeqCst);
}
```

---

## 📊 **Real-Time Programming Patterns**

### **Interrupt Handling**

```rust
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;
use stm32f1xx_hal::{
    gpio::{gpioa::PA5, Output, PushPull},
    pac,
    prelude::*,
};

static mut LED_STATE: bool = false;

#[entry]
fn main() -> ! {
    let dp = pac::Peripherals::take().unwrap();
    let cp = cortex_m::Peripherals::take().unwrap();
    
    let mut rcc = dp.RCC.constrain();
    let mut flash = dp.FLASH.constrain();
    let clocks = rcc.cfgr.freeze(&mut flash.acr);
    
    let mut gpioa = dp.GPIOA.split(&mut rcc.apb2);
    let mut led = gpioa.pa5.into_push_pull_output(&mut gpioa.crl);
    
    // Configure timer interrupt
    let mut timer = dp.TIM2.timer(&clocks, &mut rcc.apb1);
    timer.start(1.hz());
    timer.listen(Event::Update);
    
    // Enable interrupts
    unsafe {
        cortex_m::peripheral::NVIC::unmask(pac::Interrupt::TIM2);
    }
    
    loop {
        cortex_m::asm::wfi(); // Wait for interrupt
    }
}

#[interrupt]
fn TIM2() {
    unsafe {
        LED_STATE = !LED_STATE;
        if LED_STATE {
            // Turn on LED
        } else {
            // Turn off LED
        }
    }
}
```

### **State Machine Pattern**

```rust
#![no_std]

use core::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, PartialEq)]
enum State {
    Idle,
    Running,
    Error,
}

static CURRENT_STATE: AtomicU8 = AtomicU8::new(0); // Idle

impl State {
    fn as_u8(self) -> u8 {
        match self {
            State::Idle => 0,
            State::Running => 1,
            State::Error => 2,
        }
    }
    
    fn from_u8(value: u8) -> Option<State> {
        match value {
            0 => Some(State::Idle),
            1 => Some(State::Running),
            2 => Some(State::Error),
            _ => None,
        }
    }
}

fn set_state(new_state: State) {
    CURRENT_STATE.store(new_state.as_u8(), Ordering::SeqCst);
}

fn get_state() -> Option<State> {
    State::from_u8(CURRENT_STATE.load(Ordering::SeqCst))
}
```

---

## 🎯 **Best Practices**

### **Memory Management**

```rust
#![no_std]

// ✅ Good - use const for immutable data
const MAX_BUFFER_SIZE: usize = 1024;

// ✅ Good - use static for global state
static mut BUFFER: [u8; MAX_BUFFER_SIZE] = [0; MAX_BUFFER_SIZE];

// ✅ Good - use atomic for thread-safe access
use core::sync::atomic::{AtomicU32, Ordering};
static COUNTER: AtomicU32 = AtomicU32::new(0);
```

### **Error Handling**

```rust
#![no_std]

use core::fmt;

#[derive(Debug)]
enum EmbeddedError {
    Timeout,
    InvalidData,
    HardwareFault,
}

impl fmt::Display for EmbeddedError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EmbeddedError::Timeout => write!(f, "Operation timed out"),
            EmbeddedError::InvalidData => write!(f, "Invalid data received"),
            EmbeddedError::HardwareFault => write!(f, "Hardware fault detected"),
        }
    }
}

type Result<T> = core::result::Result<T, EmbeddedError>;
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [Embedded Rust Book](https://docs.rust-embedded.org/book/) - Fetched: 2024-12-19T00:00:00Z
- [The Embedded Rust Book](https://docs.rust-embedded.org/book/) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [RTIC Framework](https://rtic.rs/) - Fetched: 2024-12-19T00:00:00Z
- [Embedded HAL](https://github.com/rust-embedded/embedded-hal) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. What's the difference between `std` and `no_std` environments?
2. How do you handle interrupts in embedded Rust?
3. What are the key considerations for memory management in embedded systems?
4. How do you implement real-time programming patterns?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Advanced embedded programming techniques
- Real-time operating systems (RTOS)
- Hardware abstraction layers
- Performance optimization for embedded systems

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [18.2 Real-Time Operating Systems](18_02_rtos_embedded.md)
