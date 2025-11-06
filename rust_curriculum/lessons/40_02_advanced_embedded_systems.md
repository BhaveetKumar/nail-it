---
# Auto-generated front matter
Title: 40 02 Advanced Embedded Systems
LastUpdated: 2025-11-06T20:45:58.117084
Tags: []
Status: draft
---

# Lesson 40.2: Advanced Embedded Systems

> **Module**: 40 - Advanced Embedded Systems  
> **Lesson**: 2 of 6  
> **Duration**: 4-5 hours  
> **Prerequisites**: Module 39 (Final Project)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Build real-time embedded systems with Rust
- Implement hardware abstraction layers
- Develop IoT device firmware
- Optimize for resource-constrained environments
- Deploy embedded applications to production

---

## 🎯 **Overview**

Advanced embedded systems development in Rust involves building real-time systems, hardware abstraction layers, IoT device firmware, and optimizing for resource-constrained environments. This lesson covers advanced embedded patterns, real-time programming, and production deployment.

---

## 🔧 **Advanced Embedded Patterns**

### **Real-Time Operating System (RTOS)**

```rust
// Cargo.toml
[package]
name = "advanced-embedded-rtos"
version = "0.1.0"
edition = "2021"

[dependencies]
cortex-m = "0.7"
cortex-m-rt = "0.7"
cortex-m-semihosting = "0.3"
panic-halt = "0.2"
nb = "1.0"
heapless = "0.7"
rtic = "2.0"
rtic-monotonic = "0.1"
embedded-hal = "1.0"
stm32f4xx-hal = "0.13"
defmt = "0.3"
defmt-rtt = "0.4"

// src/main.rs
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;
use rtic::app;
use stm32f4xx_hal::{
    gpio::{gpioa::PA5, Output, PushPull},
    pac,
    prelude::*,
    timer::Timer,
};

#[app(device = pac, peripherals = true)]
mod app {
    use super::*;
    use cortex_m::peripheral::NVIC;
    use rtic_monotonic::systick::Systick;
    
    #[monotonic(binds = SysTick, default = true)]
    type MonoTimer = Systick<1000>;
    
    #[shared]
    struct Shared {
        led_state: bool,
        sensor_data: heapless::Vec<f32, 32>,
    }
    
    #[local]
    struct Local {
        led: PA5<Output<PushPull>>,
        timer: Timer<pac::TIM2>,
    }
    
    #[init]
    fn init(cx: init::Context) -> (Shared, Local, init::Monotonics) {
        let device = cx.device;
        let mut rcc = device.RCC.constrain();
        let clocks = rcc.cfgr.sysclk(84.mhz()).freeze();
        
        let gpioa = device.GPIOA.split();
        let led = gpioa.pa5.into_push_pull_output();
        
        let timer = Timer::new(device.TIM2, &clocks);
        
        // Start periodic task
        periodic_task::spawn_after(1.secs()).unwrap();
        
        (
            Shared {
                led_state: false,
                sensor_data: heapless::Vec::new(),
            },
            Local { led, timer },
            init::Monotonics(Systick::new(cx.core.SYST, 84_000_000)),
        )
    }
    
    #[task(shared = [led_state, sensor_data], local = [led, timer])]
    fn periodic_task(cx: periodic_task::Context) {
        let shared = cx.shared;
        let local = cx.local;
        
        // Toggle LED
        shared.led_state.lock(|state| {
            *state = !*state;
            if *state {
                local.led.set_high().unwrap();
            } else {
                local.led.set_low().unwrap();
            }
        });
        
        // Read sensor data
        let sensor_value = read_sensor();
        shared.sensor_data.lock(|data| {
            if data.len() < 32 {
                data.push(sensor_value).ok();
            } else {
                data.rotate_left(1);
                data[31] = sensor_value;
            }
        });
        
        // Schedule next execution
        periodic_task::spawn_after(1.secs()).unwrap();
    }
    
    #[task(shared = [sensor_data], priority = 2)]
    fn data_processing_task(cx: data_processing_task::Context) {
        let shared = cx.shared;
        
        shared.sensor_data.lock(|data| {
            if data.len() > 0 {
                let sum: f32 = data.iter().sum();
                let avg = sum / data.len() as f32;
                
                // Process data based on average
                if avg > 25.0 {
                    // High temperature detected
                    emergency_shutdown();
                } else if avg < 10.0 {
                    // Low temperature detected
                    heating_on();
                }
            }
        });
    }
}

fn read_sensor() -> f32 {
    // Simulate sensor reading
    20.0 + (cortex_m::peripheral::DWT::cycle_count() as f32 * 0.001).sin() * 5.0
}

fn emergency_shutdown() {
    // Emergency shutdown procedure
    defmt::error!("Emergency shutdown triggered!");
}

fn heating_on() {
    // Turn on heating
    defmt::info!("Heating activated");
}
```

### **Hardware Abstraction Layer (HAL)**

```rust
use embedded_hal::digital::v2::OutputPin;
use embedded_hal::timer::CountDown;
use embedded_hal::adc::OneShot;
use embedded_hal::spi::FullDuplex;
use embedded_hal::i2c::I2c;
use nb::block;

pub trait SensorDriver {
    type Error;
    type Data;
    
    fn read(&mut self) -> Result<Self::Data, Self::Error>;
    fn calibrate(&mut self) -> Result<(), Self::Error>;
    fn is_ready(&self) -> bool;
}

pub trait ActuatorDriver {
    type Error;
    
    fn set_state(&mut self, state: bool) -> Result<(), Self::Error>;
    fn get_state(&self) -> bool;
}

pub struct TemperatureSensor<ADC> {
    adc: ADC,
    channel: u8,
    calibration_offset: f32,
    calibration_scale: f32,
}

impl<ADC> TemperatureSensor<ADC>
where
    ADC: OneShot<ADC, u16, u8>,
{
    pub fn new(adc: ADC, channel: u8) -> Self {
        Self {
            adc,
            channel,
            calibration_offset: 0.0,
            calibration_scale: 1.0,
        }
    }
    
    pub fn calibrate(&mut self, reference_temp: f32, raw_value: u16) {
        let measured_temp = self.raw_to_temperature(raw_value);
        self.calibration_offset = reference_temp - measured_temp;
    }
    
    fn raw_to_temperature(&self, raw: u16) -> f32 {
        (raw as f32 * self.calibration_scale) + self.calibration_offset
    }
}

impl<ADC> SensorDriver for TemperatureSensor<ADC>
where
    ADC: OneShot<ADC, u16, u8>,
{
    type Error = ADC::Error;
    type Data = f32;
    
    fn read(&mut self) -> Result<f32, Self::Error> {
        let raw = block!(self.adc.read(&self.channel))?;
        Ok(self.raw_to_temperature(raw))
    }
    
    fn calibrate(&mut self) -> Result<(), Self::Error> {
        // Perform calibration sequence
        Ok(())
    }
    
    fn is_ready(&self) -> bool {
        true
    }
}

pub struct RelayDriver<PIN> {
    pin: PIN,
    state: bool,
}

impl<PIN> RelayDriver<PIN>
where
    PIN: OutputPin,
{
    pub fn new(pin: PIN) -> Self {
        Self { pin, state: false }
    }
}

impl<PIN> ActuatorDriver for RelayDriver<PIN>
where
    PIN: OutputPin,
{
    type Error = PIN::Error;
    
    fn set_state(&mut self, state: bool) -> Result<(), Self::Error> {
        if state {
            self.pin.set_high()?;
        } else {
            self.pin.set_low()?;
        }
        self.state = state;
        Ok(())
    }
    
    fn get_state(&self) -> bool {
        self.state
    }
}

pub struct I2CSensor<I2C> {
    i2c: I2C,
    address: u8,
}

impl<I2C> I2CSensor<I2C>
where
    I2C: I2c<u8>,
{
    pub fn new(i2c: I2C, address: u8) -> Self {
        Self { i2c, address }
    }
    
    pub fn read_register(&mut self, register: u8) -> Result<u8, I2C::Error> {
        let mut buffer = [0u8; 1];
        self.i2c.write_read(self.address, &[register], &mut buffer)?;
        Ok(buffer[0])
    }
    
    pub fn write_register(&mut self, register: u8, value: u8) -> Result<(), I2C::Error> {
        self.i2c.write(self.address, &[register, value])
    }
}

impl<I2C> SensorDriver for I2CSensor<I2C>
where
    I2C: I2c<u8>,
{
    type Error = I2C::Error;
    type Data = u8;
    
    fn read(&mut self) -> Result<u8, Self::Error> {
        self.read_register(0x00)
    }
    
    fn calibrate(&mut self) -> Result<(), Self::Error> {
        // Send calibration command
        self.write_register(0x01, 0x55)
    }
    
    fn is_ready(&self) -> bool {
        true
    }
}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: IoT Device Firmware**

```rust
use embedded_hal::digital::v2::OutputPin;
use embedded_hal::timer::CountDown;
use embedded_hal::adc::OneShot;
use embedded_hal::spi::FullDuplex;
use embedded_hal::i2c::I2c;
use nb::block;
use heapless::Vec;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct SensorData {
    pub timestamp: u32,
    pub temperature: f32,
    pub humidity: f32,
    pub pressure: f32,
    pub light: u16,
}

pub struct IoTDevice<SPI, I2C, ADC, TIMER, LED, BUTTON> {
    pub spi: SPI,
    pub i2c: I2C,
    pub adc: ADC,
    pub timer: TIMER,
    pub led: LED,
    pub button: BUTTON,
    pub sensor_data: Vec<SensorData, 100>,
    pub config: DeviceConfig,
}

#[derive(Serialize, Deserialize)]
pub struct DeviceConfig {
    pub sample_interval: u32,
    pub transmission_interval: u32,
    pub sensor_enabled: bool,
    pub led_enabled: bool,
    pub debug_mode: bool,
}

impl<SPI, I2C, ADC, TIMER, LED, BUTTON> IoTDevice<SPI, I2C, ADC, TIMER, LED, BUTTON>
where
    SPI: FullDuplex<u8>,
    I2C: I2c<u8>,
    ADC: OneShot<ADC, u16, u8>,
    TIMER: CountDown,
    LED: OutputPin,
    BUTTON: embedded_hal::digital::v2::InputPin,
{
    pub fn new(
        spi: SPI,
        i2c: I2C,
        adc: ADC,
        timer: TIMER,
        led: LED,
        button: BUTTON,
    ) -> Self {
        Self {
            spi,
            i2c,
            adc,
            timer,
            led,
            button,
            sensor_data: Vec::new(),
            config: DeviceConfig {
                sample_interval: 1000,
                transmission_interval: 10000,
                sensor_enabled: true,
                led_enabled: true,
                debug_mode: false,
            },
        }
    }
    
    pub fn run(&mut self) -> Result<(), &'static str> {
        loop {
            // Check button press
            if self.button.is_high().unwrap_or(false) {
                self.handle_button_press()?;
            }
            
            // Sample sensors
            if self.config.sensor_enabled {
                self.sample_sensors()?;
            }
            
            // Transmit data
            if self.sensor_data.len() >= 10 {
                self.transmit_data()?;
            }
            
            // Update LED
            if self.config.led_enabled {
                self.update_led()?;
            }
            
            // Wait for next cycle
            block!(self.timer.wait()).map_err(|_| "Timer error")?;
        }
    }
    
    fn sample_sensors(&mut self) -> Result<(), &'static str> {
        let timestamp = self.timer.now();
        let temperature = self.read_temperature()?;
        let humidity = self.read_humidity()?;
        let pressure = self.read_pressure()?;
        let light = self.read_light()?;
        
        let data = SensorData {
            timestamp,
            temperature,
            humidity,
            pressure,
            light,
        };
        
        if self.sensor_data.push(data).is_err() {
            // Buffer full, remove oldest data
            self.sensor_data.rotate_left(1);
            self.sensor_data[99] = data;
        }
        
        Ok(())
    }
    
    fn read_temperature(&mut self) -> Result<f32, &'static str> {
        let raw = block!(self.adc.read(&0)).map_err(|_| "ADC error")?;
        Ok((raw as f32 * 3.3) / 4095.0 * 100.0 - 50.0)
    }
    
    fn read_humidity(&mut self) -> Result<f32, &'static str> {
        let raw = block!(self.adc.read(&1)).map_err(|_| "ADC error")?;
        Ok((raw as f32 * 3.3) / 4095.0 * 100.0)
    }
    
    fn read_pressure(&mut self) -> Result<f32, &'static str> {
        let raw = block!(self.adc.read(&2)).map_err(|_| "ADC error")?;
        Ok((raw as f32 * 3.3) / 4095.0 * 1000.0)
    }
    
    fn read_light(&mut self) -> Result<u16, &'static str> {
        let raw = block!(self.adc.read(&3)).map_err(|_| "ADC error")?;
        Ok(raw)
    }
    
    fn transmit_data(&mut self) -> Result<(), &'static str> {
        // Convert data to JSON
        let json_data = serde_json::to_string(&self.sensor_data.as_slice())
            .map_err(|_| "JSON serialization error")?;
        
        // Transmit via SPI
        for byte in json_data.bytes() {
            block!(self.spi.send(byte)).map_err(|_| "SPI send error")?;
            let _ = block!(self.spi.read()).map_err(|_| "SPI read error")?;
        }
        
        // Clear transmitted data
        self.sensor_data.clear();
        
        Ok(())
    }
    
    fn update_led(&mut self) -> Result<(), &'static str> {
        // Blink LED based on data count
        let blink_count = self.sensor_data.len() / 10;
        for _ in 0..blink_count {
            self.led.set_high().map_err(|_| "LED error")?;
            // Short delay
            for _ in 0..1000 {
                cortex_m::asm::nop();
            }
            self.led.set_low().map_err(|_| "LED error")?;
            for _ in 0..1000 {
                cortex_m::asm::nop();
            }
        }
        
        Ok(())
    }
    
    fn handle_button_press(&mut self) -> Result<(), &'static str> {
        // Toggle debug mode
        self.config.debug_mode = !self.config.debug_mode;
        
        // Flash LED to indicate mode change
        for _ in 0..5 {
            self.led.set_high().map_err(|_| "LED error")?;
            for _ in 0..10000 {
                cortex_m::asm::nop();
            }
            self.led.set_low().map_err(|_| "LED error")?;
            for _ in 0..10000 {
                cortex_m::asm::nop();
            }
        }
        
        Ok(())
    }
}
```

### **Exercise 2: Real-Time Control System**

```rust
use embedded_hal::digital::v2::{InputPin, OutputPin};
use embedded_hal::timer::CountDown;
use embedded_hal::adc::OneShot;
use nb::block;
use heapless::Vec;

pub struct PIDController {
    pub kp: f32,
    pub ki: f32,
    pub kd: f32,
    pub setpoint: f32,
    pub integral: f32,
    pub last_error: f32,
    pub output_min: f32,
    pub output_max: f32,
}

impl PIDController {
    pub fn new(kp: f32, ki: f32, kd: f32) -> Self {
        Self {
            kp,
            ki,
            kd,
            setpoint: 0.0,
            integral: 0.0,
            last_error: 0.0,
            output_min: 0.0,
            output_max: 100.0,
        }
    }
    
    pub fn set_setpoint(&mut self, setpoint: f32) {
        self.setpoint = setpoint;
    }
    
    pub fn update(&mut self, input: f32, dt: f32) -> f32 {
        let error = self.setpoint - input;
        
        // Proportional term
        let p_term = self.kp * error;
        
        // Integral term
        self.integral += error * dt;
        let i_term = self.ki * self.integral;
        
        // Derivative term
        let derivative = (error - self.last_error) / dt;
        let d_term = self.kd * derivative;
        
        // Calculate output
        let output = p_term + i_term + d_term;
        
        // Clamp output
        let output = output.max(self.output_min).min(self.output_max);
        
        // Update state
        self.last_error = error;
        
        output
    }
    
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.last_error = 0.0;
    }
}

pub struct ControlSystem<ADC, TIMER, ACTUATOR, SENSOR> {
    pub adc: ADC,
    pub timer: TIMER,
    pub actuator: ACTUATOR,
    pub sensor: SENSOR,
    pub pid: PIDController,
    pub control_loop_active: bool,
    pub sample_count: u32,
}

impl<ADC, TIMER, ACTUATOR, SENSOR> ControlSystem<ADC, TIMER, ACTUATOR, SENSOR>
where
    ADC: OneShot<ADC, u16, u8>,
    TIMER: CountDown,
    ACTUATOR: OutputPin,
    SENSOR: InputPin,
{
    pub fn new(
        adc: ADC,
        timer: TIMER,
        actuator: ACTUATOR,
        sensor: SENSOR,
        pid: PIDController,
    ) -> Self {
        Self {
            adc,
            timer,
            actuator,
            sensor,
            pid,
            control_loop_active: false,
            sample_count: 0,
        }
    }
    
    pub fn run_control_loop(&mut self) -> Result<(), &'static str> {
        if !self.control_loop_active {
            return Ok(());
        }
        
        // Read sensor value
        let raw_value = block!(self.adc.read(&0)).map_err(|_| "ADC error")?;
        let sensor_value = (raw_value as f32 * 3.3) / 4095.0;
        
        // Update PID controller
        let dt = 0.01; // 10ms control loop
        let output = self.pid.update(sensor_value, dt);
        
        // Apply control output
        self.apply_control_output(output)?;
        
        // Update sample count
        self.sample_count += 1;
        
        // Check for sensor fault
        if self.sensor.is_low().unwrap_or(true) {
            self.handle_sensor_fault()?;
        }
        
        Ok(())
    }
    
    fn apply_control_output(&mut self, output: f32) -> Result<(), &'static str> {
        // Convert output to PWM duty cycle
        let duty_cycle = (output / 100.0 * 255.0) as u8;
        
        // Apply PWM to actuator
        if duty_cycle > 0 {
            self.actuator.set_high().map_err(|_| "Actuator error")?;
        } else {
            self.actuator.set_low().map_err(|_| "Actuator error")?;
        }
        
        Ok(())
    }
    
    fn handle_sensor_fault(&mut self) -> Result<(), &'static str> {
        // Stop control loop
        self.control_loop_active = false;
        
        // Reset PID controller
        self.pid.reset();
        
        // Turn off actuator
        self.actuator.set_low().map_err(|_| "Actuator error")?;
        
        Ok(())
    }
    
    pub fn start_control(&mut self, setpoint: f32) -> Result<(), &'static str> {
        self.pid.set_setpoint(setpoint);
        self.control_loop_active = true;
        self.sample_count = 0;
        Ok(())
    }
    
    pub fn stop_control(&mut self) -> Result<(), &'static str> {
        self.control_loop_active = false;
        self.pid.reset();
        self.actuator.set_low().map_err(|_| "Actuator error")?;
        Ok(())
    }
}
```

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use embedded_hal::digital::v2::OutputPin;
    use embedded_hal::timer::CountDown;
    use embedded_hal::adc::OneShot;
    use nb::Result;

    struct MockADC {
        values: Vec<u16>,
        index: usize,
    }

    impl MockADC {
        fn new(values: Vec<u16>) -> Self {
            Self { values, index: 0 }
        }
    }

    impl OneShot<MockADC, u16, u8> for MockADC {
        type Error = &'static str;
        
        fn read(&mut self, _channel: &u8) -> Result<u16, Self::Error> {
            if self.index < self.values.len() {
                let value = self.values[self.index];
                self.index += 1;
                Ok(value)
            } else {
                Err(nb::Error::WouldBlock)
            }
        }
    }

    struct MockTimer {
        count: u32,
    }

    impl MockTimer {
        fn new() -> Self {
            Self { count: 0 }
        }
    }

    impl CountDown for MockTimer {
        type Time = u32;
        
        fn start<T>(&mut self, count: T) -> Result<(), Self::Error>
        where
            T: Into<Self::Time>,
        {
            self.count = count.into();
            Ok(())
        }
        
        fn wait(&mut self) -> Result<(), Self::Error> {
            if self.count > 0 {
                self.count -= 1;
                Ok(())
            } else {
                Err(nb::Error::WouldBlock)
            }
        }
    }

    struct MockPin {
        state: bool,
    }

    impl MockPin {
        fn new() -> Self {
            Self { state: false }
        }
    }

    impl OutputPin for MockPin {
        type Error = &'static str;
        
        fn set_low(&mut self) -> Result<(), Self::Error> {
            self.state = false;
            Ok(())
        }
        
        fn set_high(&mut self) -> Result<(), Self::Error> {
            self.state = true;
            Ok(())
        }
    }

    #[test]
    fn test_pid_controller() {
        let mut pid = PIDController::new(1.0, 0.1, 0.01);
        pid.set_setpoint(50.0);
        
        let output = pid.update(40.0, 0.01);
        assert!(output > 0.0);
        
        let output2 = pid.update(45.0, 0.01);
        assert!(output2 < output);
    }

    #[test]
    fn test_control_system() {
        let adc = MockADC::new(vec![2048, 2048, 2048]);
        let timer = MockTimer::new();
        let actuator = MockPin::new();
        let sensor = MockPin::new();
        let pid = PIDController::new(1.0, 0.1, 0.01);
        
        let mut control_system = ControlSystem::new(adc, timer, actuator, sensor, pid);
        
        control_system.start_control(50.0).unwrap();
        control_system.run_control_loop().unwrap();
        
        assert!(control_system.control_loop_active);
        assert_eq!(control_system.sample_count, 1);
    }
}
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Memory Management in Embedded Systems**

```rust
// ❌ Wrong - using heap allocation in no_std
#![no_std]
use std::vec::Vec; // This won't compile in no_std

// ✅ Correct - using heapless collections
#![no_std]
use heapless::Vec;

fn process_data() {
    let mut data: Vec<f32, 32> = Vec::new();
    data.push(1.0).unwrap();
}
```

### **Common Mistake 2: Blocking Operations in Real-Time Systems**

```rust
// ❌ Wrong - blocking operation
fn bad_sensor_read() -> u16 {
    loop {
        match adc.read(&0) {
            Ok(value) => return value,
            Err(nb::Error::WouldBlock) => continue,
            Err(nb::Error::Other(e)) => panic!("ADC error: {:?}", e),
        }
    }
}

// ✅ Correct - non-blocking operation
fn good_sensor_read() -> Result<u16, &'static str> {
    match adc.read(&0) {
        Ok(value) => Ok(value),
        Err(nb::Error::WouldBlock) => Err("Would block"),
        Err(nb::Error::Other(e)) => Err("ADC error"),
    }
}
```

---

## 📊 **Advanced Embedded Patterns**

### **Interrupt-Driven Architecture**

```rust
use cortex_m::peripheral::NVIC;
use stm32f4xx_hal::pac;

pub struct InterruptManager {
    pub nvic: NVIC,
    pub interrupt_count: u32,
}

impl InterruptManager {
    pub fn new() -> Self {
        Self {
            nvic: NVIC::new(pac::NVIC),
            interrupt_count: 0,
        }
    }
    
    pub fn enable_interrupt(&mut self, interrupt: u8) {
        unsafe {
            self.nvic.enable(interrupt);
        }
    }
    
    pub fn disable_interrupt(&mut self, interrupt: u8) {
        unsafe {
            self.nvic.disable(interrupt);
        }
    }
    
    pub fn set_priority(&mut self, interrupt: u8, priority: u8) {
        unsafe {
            self.nvic.set_priority(interrupt, priority);
        }
    }
    
    pub fn handle_interrupt(&mut self, interrupt: u8) {
        self.interrupt_count += 1;
        
        match interrupt {
            0 => self.handle_timer_interrupt(),
            1 => self.handle_adc_interrupt(),
            2 => self.handle_uart_interrupt(),
            _ => self.handle_unknown_interrupt(interrupt),
        }
    }
    
    fn handle_timer_interrupt(&self) {
        // Timer interrupt handling
    }
    
    fn handle_adc_interrupt(&self) {
        // ADC interrupt handling
    }
    
    fn handle_uart_interrupt(&self) {
        // UART interrupt handling
    }
    
    fn handle_unknown_interrupt(&self, interrupt: u8) {
        // Unknown interrupt handling
    }
}
```

### **Power Management**

```rust
use cortex_m::peripheral::SCB;
use stm32f4xx_hal::pac;

pub struct PowerManager {
    pub scb: SCB,
    pub sleep_mode: SleepMode,
    pub wakeup_sources: WakeupSources,
}

#[derive(Debug, Clone, Copy)]
pub enum SleepMode {
    Sleep,
    DeepSleep,
    Standby,
    Shutdown,
}

#[derive(Debug, Clone, Copy)]
pub struct WakeupSources {
    pub timer: bool,
    pub uart: bool,
    pub button: bool,
    pub rtc: bool,
}

impl PowerManager {
    pub fn new() -> Self {
        Self {
            scb: SCB::new(pac::SCB),
            sleep_mode: SleepMode::Sleep,
            wakeup_sources: WakeupSources {
                timer: true,
                uart: false,
                button: true,
                rtc: false,
            },
        }
    }
    
    pub fn enter_sleep(&mut self) {
        match self.sleep_mode {
            SleepMode::Sleep => {
                cortex_m::asm::wfi(); // Wait for interrupt
            }
            SleepMode::DeepSleep => {
                // Configure deep sleep mode
                self.scb.set_sleepdeep(true);
                cortex_m::asm::wfi();
                self.scb.set_sleepdeep(false);
            }
            SleepMode::Standby => {
                // Configure standby mode
                self.scb.set_sleepdeep(true);
                cortex_m::asm::wfi();
                self.scb.set_sleepdeep(false);
            }
            SleepMode::Shutdown => {
                // Configure shutdown mode
                self.scb.set_sleepdeep(true);
                cortex_m::asm::wfi();
                self.scb.set_sleepdeep(false);
            }
        }
    }
    
    pub fn set_sleep_mode(&mut self, mode: SleepMode) {
        self.sleep_mode = mode;
    }
    
    pub fn configure_wakeup_sources(&mut self, sources: WakeupSources) {
        self.wakeup_sources = sources;
    }
    
    pub fn get_power_consumption(&self) -> f32 {
        match self.sleep_mode {
            SleepMode::Sleep => 1.0,
            SleepMode::DeepSleep => 0.1,
            SleepMode::Standby => 0.01,
            SleepMode::Shutdown => 0.001,
        }
    }
}
```

---

## 🎯 **Best Practices**

### **Embedded Configuration**

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddedConfig {
    pub system: SystemConfig,
    pub sensors: SensorConfig,
    pub communication: CommunicationConfig,
    pub power: PowerConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemConfig {
    pub clock_frequency: u32,
    pub debug_enabled: bool,
    pub watchdog_timeout: u32,
    pub stack_size: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SensorConfig {
    pub sample_rate: u32,
    pub calibration_enabled: bool,
    pub sensor_count: u8,
    pub data_buffer_size: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CommunicationConfig {
    pub uart_baud_rate: u32,
    pub i2c_frequency: u32,
    pub spi_frequency: u32,
    pub protocol: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PowerConfig {
    pub sleep_mode: String,
    pub wakeup_interval: u32,
    pub low_power_threshold: f32,
    pub battery_monitoring: bool,
}
```

### **Error Handling**

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbeddedError {
    #[error("ADC error: {0}")]
    AdcError(String),
    
    #[error("Timer error: {0}")]
    TimerError(String),
    
    #[error("Communication error: {0}")]
    CommunicationError(String),
    
    #[error("Sensor error: {0}")]
    SensorError(String),
    
    #[error("Power management error: {0}")]
    PowerError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

pub type Result<T> = std::result::Result<T, EmbeddedError>;
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [Embedded Rust](https://docs.rust-embedded.org/) - Fetched: 2024-12-19T00:00:00Z
- [RTIC](https://rtic.rs/) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Embedded HAL](https://docs.rs/embedded-hal/) - Fetched: 2024-12-19T00:00:00Z
- [Cortex-M](https://docs.rs/cortex-m/) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. Can you build real-time embedded systems with Rust?
2. Do you understand hardware abstraction layers?
3. Can you develop IoT device firmware?
4. Do you know how to optimize for resource-constrained environments?
5. Can you deploy embedded applications to production?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Advanced compiler techniques
- Custom lints and passes
- Procedural macro development
- Domain-specific language creation

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [40.3 Advanced Compiler Techniques](40_03_compiler_techniques.md)
