# Go Fundamentals

## Table of Contents

1. [Overview](#overview)
2. [Basic Syntax](#basic-syntax)
3. [Data Types](#data-types)
4. [Control Structures](#control-structures)
5. [Functions](#functions)
6. [Structs and Interfaces](#structs-and-interfaces)
7. [Concurrency](#concurrency)
8. [Error Handling](#error-handling)
9. [Packages and Modules](#packages-and-modules)
10. [Best Practices](#best-practices)

## Overview

### Learning Objectives

- Master Go syntax and basic concepts
- Understand Go's type system and memory management
- Learn Go's concurrency model with goroutines
- Apply Go best practices and idioms
- Build real-world applications with Go

### What is Go?

Go (Golang) is a statically typed, compiled programming language designed for simplicity, efficiency, and concurrency. It's widely used for backend services, microservices, and system programming.

## Basic Syntax

### 1. Hello World and Basic Structure

```go
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
    
    // Variables
    var name string = "Go"
    age := 25 // Type inference
    
    fmt.Printf("Name: %s, Age: %d\n", name, age)
}
```

### 2. Variables and Constants

```go
package main

import "fmt"

func main() {
    // Variable declarations
    var a int = 10
    var b string = "hello"
    var c bool = true
    
    // Short declaration
    x := 42
    y := "world"
    
    // Multiple declarations
    var (
        name    string = "John"
        age     int    = 30
        isActive bool  = true
    )
    
    // Constants
    const pi = 3.14159
    const (
        StatusOK = 200
        StatusNotFound = 404
    )
    
    fmt.Println(a, b, c, x, y)
    fmt.Println(name, age, isActive)
    fmt.Println(pi, StatusOK)
}
```

## Data Types

### 1. Basic Types

```go
package main

import "fmt"

func main() {
    // Numeric types
    var i int = 42
    var f float64 = 3.14
    var c complex128 = 1 + 2i
    
    // String
    var s string = "Hello, Go!"
    
    // Boolean
    var b bool = true
    
    // Arrays
    var arr [5]int = [5]int{1, 2, 3, 4, 5}
    
    // Slices
    var slice []int = []int{1, 2, 3, 4, 5}
    slice = append(slice, 6)
    
    // Maps
    var m map[string]int = make(map[string]int)
    m["key"] = 42
    
    fmt.Printf("int: %d, float: %.2f, complex: %v\n", i, f, c)
    fmt.Printf("string: %s, bool: %t\n", s, b)
    fmt.Printf("array: %v, slice: %v, map: %v\n", arr, slice, m)
}
```

### 2. Pointers and Memory

```go
package main

import "fmt"

func main() {
    x := 42
    p := &x // Pointer to x
    
    fmt.Printf("Value: %d, Address: %p\n", x, p)
    fmt.Printf("Value through pointer: %d\n", *p)
    
    *p = 100 // Change value through pointer
    fmt.Printf("New value: %d\n", x)
    
    // Pointer to pointer
    pp := &p
    fmt.Printf("Pointer to pointer: %p\n", pp)
    fmt.Printf("Value through double pointer: %d\n", **pp)
}
```

## Control Structures

### 1. If-Else and Switch

```go
package main

import "fmt"

func main() {
    // If-else
    x := 10
    if x > 5 {
        fmt.Println("x is greater than 5")
    } else if x == 5 {
        fmt.Println("x equals 5")
    } else {
        fmt.Println("x is less than 5")
    }
    
    // Switch
    day := "Monday"
    switch day {
    case "Monday":
        fmt.Println("Start of work week")
    case "Friday":
        fmt.Println("End of work week")
    case "Saturday", "Sunday":
        fmt.Println("Weekend")
    default:
        fmt.Println("Mid week")
    }
    
    // Switch with no expression
    switch {
    case x < 0:
        fmt.Println("Negative")
    case x == 0:
        fmt.Println("Zero")
    default:
        fmt.Println("Positive")
    }
}
```

### 2. Loops

```go
package main

import "fmt"

func main() {
    // For loop
    for i := 0; i < 5; i++ {
        fmt.Printf("i: %d\n", i)
    }
    
    // While-like loop
    j := 0
    for j < 5 {
        fmt.Printf("j: %d\n", j)
        j++
    }
    
    // Infinite loop with break
    k := 0
    for {
        if k >= 3 {
            break
        }
        fmt.Printf("k: %d\n", k)
        k++
    }
    
    // Range over slice
    numbers := []int{1, 2, 3, 4, 5}
    for index, value := range numbers {
        fmt.Printf("Index: %d, Value: %d\n", index, value)
    }
    
    // Range over map
    colors := map[string]string{
        "red":   "#FF0000",
        "green": "#00FF00",
        "blue":  "#0000FF",
    }
    for key, value := range colors {
        fmt.Printf("Color: %s, Hex: %s\n", key, value)
    }
}
```

## Functions

### 1. Basic Functions

```go
package main

import "fmt"

// Basic function
func add(a, b int) int {
    return a + b
}

// Multiple return values
func divide(a, b int) (int, error) {
    if b == 0 {
        return 0, fmt.Errorf("division by zero")
    }
    return a / b, nil
}

// Named return values
func swap(a, b int) (x, y int) {
    x = b
    y = a
    return // naked return
}

// Variadic function
func sum(numbers ...int) int {
    total := 0
    for _, num := range numbers {
        total += num
    }
    return total
}

// Function as parameter
func apply(f func(int) int, x int) int {
    return f(x)
}

// Anonymous function
func main() {
    result := add(5, 3)
    fmt.Println("5 + 3 =", result)
    
    quotient, err := divide(10, 2)
    if err != nil {
        fmt.Println("Error:", err)
    } else {
        fmt.Println("10 / 2 =", quotient)
    }
    
    a, b := swap(1, 2)
    fmt.Printf("Swapped: %d, %d\n", a, b)
    
    total := sum(1, 2, 3, 4, 5)
    fmt.Println("Sum:", total)
    
    // Anonymous function
    square := func(x int) int {
        return x * x
    }
    fmt.Println("Square of 5:", square(5))
    
    // Function as parameter
    result = apply(square, 4)
    fmt.Println("Applied square to 4:", result)
}
```

## Structs and Interfaces

### 1. Structs

```go
package main

import "fmt"

// Basic struct
type Person struct {
    Name string
    Age  int
}

// Method on struct
func (p Person) Greet() string {
    return fmt.Sprintf("Hello, I'm %s and I'm %d years old", p.Name, p.Age)
}

// Pointer receiver method
func (p *Person) HaveBirthday() {
    p.Age++
}

// Embedded struct
type Employee struct {
    Person
    ID       int
    Position string
}

// Method on embedded struct
func (e Employee) Work() string {
    return fmt.Sprintf("%s is working as %s", e.Name, e.Position)
}

func main() {
    // Create struct instances
    person := Person{Name: "Alice", Age: 30}
    fmt.Println(person.Greet())
    
    person.HaveBirthday()
    fmt.Println("After birthday:", person.Age)
    
    // Embedded struct
    employee := Employee{
        Person:   Person{Name: "Bob", Age: 25},
        ID:       123,
        Position: "Developer",
    }
    
    fmt.Println(employee.Greet()) // Can call Person methods
    fmt.Println(employee.Work())
}
```

### 2. Interfaces

```go
package main

import "fmt"

// Interface definition
type Shape interface {
    Area() float64
    Perimeter() float64
}

// Rectangle implements Shape
type Rectangle struct {
    Width  float64
    Height float64
}

func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

func (r Rectangle) Perimeter() float64 {
    return 2 * (r.Width + r.Height)
}

// Circle implements Shape
type Circle struct {
    Radius float64
}

func (c Circle) Area() float64 {
    return 3.14159 * c.Radius * c.Radius
}

func (c Circle) Perimeter() float64 {
    return 2 * 3.14159 * c.Radius
}

// Function that works with any Shape
func printShapeInfo(s Shape) {
    fmt.Printf("Area: %.2f, Perimeter: %.2f\n", s.Area(), s.Perimeter())
}

func main() {
    rect := Rectangle{Width: 5, Height: 3}
    circle := Circle{Radius: 4}
    
    printShapeInfo(rect)
    printShapeInfo(circle)
    
    // Interface slice
    shapes := []Shape{rect, circle}
    for _, shape := range shapes {
        printShapeInfo(shape)
    }
}
```

## Concurrency

### 1. Goroutines and Channels

```go
package main

import (
    "fmt"
    "time"
)

// Simple goroutine
func sayHello(name string) {
    for i := 0; i < 3; i++ {
        fmt.Printf("Hello %s! (%d)\n", name, i+1)
        time.Sleep(100 * time.Millisecond)
    }
}

// Channel communication
func producer(ch chan<- int) {
    for i := 0; i < 5; i++ {
        ch <- i
        fmt.Printf("Produced: %d\n", i)
    }
    close(ch)
}

func consumer(ch <-chan int) {
    for value := range ch {
        fmt.Printf("Consumed: %d\n", value)
        time.Sleep(200 * time.Millisecond)
    }
}

// Select statement
func selectExample() {
    ch1 := make(chan string)
    ch2 := make(chan string)
    
    go func() {
        time.Sleep(1 * time.Second)
        ch1 <- "from ch1"
    }()
    
    go func() {
        time.Sleep(2 * time.Second)
        ch2 <- "from ch2"
    }()
    
    for i := 0; i < 2; i++ {
        select {
        case msg1 := <-ch1:
            fmt.Println("Received:", msg1)
        case msg2 := <-ch2:
            fmt.Println("Received:", msg2)
        case <-time.After(3 * time.Second):
            fmt.Println("Timeout!")
        }
    }
}

func main() {
    // Basic goroutines
    go sayHello("Alice")
    go sayHello("Bob")
    
    // Channel communication
    ch := make(chan int)
    go producer(ch)
    go consumer(ch)
    
    // Wait for goroutines
    time.Sleep(2 * time.Second)
    
    // Select example
    selectExample()
}
```

## Error Handling

### 1. Error Patterns

```go
package main

import (
    "errors"
    "fmt"
    "strconv"
)

// Custom error type
type ValidationError struct {
    Field   string
    Message string
}

func (e ValidationError) Error() string {
    return fmt.Sprintf("validation error in %s: %s", e.Field, e.Message)
}

// Function that returns error
func divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}

// Function with custom error
func validateAge(age int) error {
    if age < 0 {
        return ValidationError{
            Field:   "age",
            Message: "age cannot be negative",
        }
    }
    if age > 150 {
        return ValidationError{
            Field:   "age",
            Message: "age cannot be greater than 150",
        }
    }
    return nil
}

// Error wrapping
func parseNumber(s string) (int, error) {
    num, err := strconv.Atoi(s)
    if err != nil {
        return 0, fmt.Errorf("failed to parse number '%s': %w", s, err)
    }
    return num, nil
}

func main() {
    // Basic error handling
    result, err := divide(10, 2)
    if err != nil {
        fmt.Println("Error:", err)
    } else {
        fmt.Println("Result:", result)
    }
    
    // Custom error
    err = validateAge(-5)
    if err != nil {
        fmt.Println("Validation error:", err)
    }
    
    // Error wrapping
    num, err := parseNumber("abc")
    if err != nil {
        fmt.Println("Parse error:", err)
    } else {
        fmt.Println("Parsed number:", num)
    }
}
```

## Packages and Modules

### 1. Package Structure

```go
// mathutils/mathutils.go
package mathutils

// Exported function (capitalized)
func Add(a, b int) int {
    return a + b
}

// Unexported function (lowercase)
func multiply(a, b int) int {
    return a * b
}

// Exported constant
const Pi = 3.14159

// Exported variable
var Version = "1.0.0"
```

### 2. Module Usage

```go
// go.mod
module myapp

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/lib/pq v1.10.9
)

// main.go
package main

import (
    "fmt"
    "myapp/mathutils"
)

func main() {
    result := mathutils.Add(5, 3)
    fmt.Println("Result:", result)
    fmt.Println("Pi:", mathutils.Pi)
    fmt.Println("Version:", mathutils.Version)
}
```

## Best Practices

### 1. Code Organization

```go
package main

import (
    "fmt"
    "log"
    "os"
)

// Constants at package level
const (
    DefaultPort = "8080"
    MaxRetries  = 3
)

// Global variables (use sparingly)
var (
    logger *log.Logger
)

// Init function
func init() {
    logger = log.New(os.Stdout, "APP: ", log.LstdFlags)
}

// Main function
func main() {
    logger.Println("Application starting...")
    
    port := os.Getenv("PORT")
    if port == "" {
        port = DefaultPort
    }
    
    fmt.Printf("Server starting on port %s\n", port)
}
```

### 2. Error Handling Best Practices

```go
package main

import (
    "errors"
    "fmt"
    "io"
    "os"
)

// Use sentinel errors for common cases
var (
    ErrNotFound = errors.New("not found")
    ErrInvalid  = errors.New("invalid input")
)

// Function with proper error handling
func readFile(filename string) ([]byte, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, fmt.Errorf("failed to open file %s: %w", filename, err)
    }
    defer file.Close()
    
    data, err := io.ReadAll(file)
    if err != nil {
        return nil, fmt.Errorf("failed to read file %s: %w", filename, err)
    }
    
    return data, nil
}

func main() {
    data, err := readFile("example.txt")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    
    fmt.Printf("File content: %s\n", string(data))
}
```

## Follow-up Questions

### 1. Basic Concepts
**Q: What's the difference between `var` and `:=` in Go?**
A: `var` declares a variable with explicit type, while `:=` declares and initializes a variable with type inference. Use `var` for zero values, `:=` for initialization.

### 2. Concurrency
**Q: What's the difference between goroutines and threads?**
A: Goroutines are lightweight, managed by the Go runtime, and multiplexed onto OS threads. They're much cheaper to create and switch between than traditional threads.

### 3. Error Handling
**Q: Why does Go use explicit error handling instead of exceptions?**
A: Explicit error handling makes errors visible in the code, forces developers to handle them, and prevents silent failures. It's more predictable and easier to debug.

## Sources

### Books
- **The Go Programming Language** by Alan Donovan and Brian Kernighan
- **Effective Go** - Official Go documentation
- **Go in Action** by William Kennedy

### Online Resources
- **Go Documentation** - https://golang.org/doc/
- **Go by Example** - https://gobyexample.com/
- **Go Playground** - https://play.golang.org/

## Projects

### 1. CLI Tool
**Objective**: Build a command-line tool
**Requirements**: Flag parsing, file I/O, error handling
**Deliverables**: Working CLI application

### 2. Web Server
**Objective**: Create a simple HTTP server
**Requirements**: HTTP handling, JSON parsing, routing
**Deliverables**: REST API server

### 3. Concurrent Program
**Objective**: Build a program using goroutines and channels
**Requirements**: Concurrency patterns, synchronization
**Deliverables**: Multi-threaded application

---

**Next**: [Node.js Fundamentals](./nodejs-fundamentals.md) | **Previous**: [Mathematics](../mathematics/README.md) | **Up**: [Phase 0](../README.md)

