---
# Auto-generated front matter
Title: Golang Complete Beginner Guide
LastUpdated: 2025-11-06T20:45:58.761046
Tags: []
Status: draft
---

# 🚀 Go (Golang) Complete Beginner's Guide

> **Learn Go from scratch with comprehensive examples and hands-on projects**

## 📚 Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Syntax](#basic-syntax)
3. [Data Types](#data-types)
4. [Control Structures](#control-structures)
5. [Functions](#functions)
6. [Structs and Methods](#structs-and-methods)
7. [Interfaces](#interfaces)
8. [Concurrency](#concurrency)
9. [Error Handling](#error-handling)
10. [Packages and Modules](#packages-and-modules)
11. [Hands-on Projects](#hands-on-projects)
12. [Practice Exercises](#practice-exercises)

---

## 🚀 Getting Started

### What is Go?

Go (also known as Golang) is a programming language developed by Google in 2009. It's designed for:
- **Simplicity**: Easy to learn and read
- **Performance**: Fast compilation and execution
- **Concurrency**: Built-in support for concurrent programming
- **Reliability**: Strong typing and garbage collection

### Why Learn Go?

- **Backend Development**: Perfect for web services and APIs
- **Microservices**: Excellent for building distributed systems
- **DevOps Tools**: Many popular tools are written in Go
- **Cloud Native**: Great for containerized applications
- **High Performance**: Fast execution and low memory usage

### Installing Go

#### Windows
1. Download from https://golang.org/dl/
2. Run the installer
3. Verify installation: `go version`

#### macOS
```bash
# Using Homebrew
brew install go

# Verify installation
go version
```

#### Linux
```bash
# Download and install
wget https://golang.org/dl/go1.21.0.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz

# Add to PATH
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc

# Verify installation
go version
```

### Your First Go Program

Create a file called `hello.go`:

```go
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
```

Run it:
```bash
go run hello.go
```

**Output:**
```
Hello, World!
```

### Understanding the Code

```go
package main
```
- Every Go program starts with a package declaration
- `main` package is special - it creates an executable

```go
import "fmt"
```
- Import the `fmt` package for formatted I/O
- `fmt` provides functions like `Println`, `Printf`, etc.

```go
func main() {
    // code here
}
```
- `main` function is the entry point of the program
- Every executable Go program must have a `main` function

---

## 📝 Basic Syntax

### Variables

#### Declaration and Initialization

```go
package main

import "fmt"

func main() {
    // Method 1: Declare then assign
    var name string
    name = "John"
    
    // Method 2: Declare and initialize
    var age int = 25
    
    // Method 3: Type inference
    var city = "New York"
    
    // Method 4: Short declaration (most common)
    country := "USA"
    
    fmt.Println(name, age, city, country)
}
```

#### Multiple Variables

```go
package main

import "fmt"

func main() {
    // Multiple variables of same type
    var a, b, c int = 1, 2, 3
    
    // Multiple variables of different types
    var (
        name    string = "Alice"
        age     int    = 30
        salary  float64 = 50000.0
        married bool   = true
    )
    
    // Short declaration for multiple variables
    x, y := 10, 20
    
    fmt.Println(a, b, c)
    fmt.Println(name, age, salary, married)
    fmt.Println(x, y)
}
```

### Constants

```go
package main

import "fmt"

func main() {
    // Single constant
    const pi = 3.14159
    
    // Multiple constants
    const (
        statusOK = 200
        statusNotFound = 404
        statusError = 500
    )
    
    // Typed constant
    const name string = "Go"
    
    fmt.Println(pi)
    fmt.Println(statusOK, statusNotFound, statusError)
    fmt.Println(name)
}
```

---

## 🔢 Data Types

### Basic Types

```go
package main

import "fmt"

func main() {
    // Integer types
    var a int = 42           // Platform-dependent size
    var b int8 = 127         // 8-bit integer (-128 to 127)
    var c int16 = 32767      // 16-bit integer
    var d int32 = 2147483647 // 32-bit integer
    var e int64 = 9223372036854775807 // 64-bit integer
    
    // Unsigned integers
    var f uint = 42          // Unsigned integer
    var g uint8 = 255        // 8-bit unsigned (0 to 255)
    
    // Floating point
    var h float32 = 3.14     // 32-bit floating point
    var i float64 = 3.14159  // 64-bit floating point
    
    // Boolean
    var j bool = true
    
    // String
    var k string = "Hello, Go!"
    
    // Complex numbers
    var l complex64 = 1 + 2i
    var m complex128 = 1 + 2i
    
    fmt.Printf("Integers: %d, %d, %d, %d, %d\n", a, b, c, d, e)
    fmt.Printf("Unsigned: %d, %d\n", f, g)
    fmt.Printf("Floats: %f, %f\n", h, i)
    fmt.Printf("Boolean: %t\n", j)
    fmt.Printf("String: %s\n", k)
    fmt.Printf("Complex: %v, %v\n", l, m)
}
```

### String Operations

```go
package main

import (
    "fmt"
    "strings"
)

func main() {
    // String concatenation
    firstName := "John"
    lastName := "Doe"
    fullName := firstName + " " + lastName
    fmt.Println("Full name:", fullName)
    
    // String length
    fmt.Println("Length:", len(fullName))
    
    // String comparison
    str1 := "hello"
    str2 := "world"
    fmt.Println("Equal:", str1 == str2)
    
    // String methods
    text := "Hello, World!"
    fmt.Println("Uppercase:", strings.ToUpper(text))
    fmt.Println("Lowercase:", strings.ToLower(text))
    fmt.Println("Contains 'World':", strings.Contains(text, "World"))
    fmt.Println("Replace:", strings.Replace(text, "World", "Go", 1))
    
    // String slicing
    fmt.Println("First 5 chars:", text[:5])
    fmt.Println("Last 5 chars:", text[len(text)-5:])
    fmt.Println("Middle chars:", text[2:8])
}
```

### Arrays and Slices

```go
package main

import "fmt"

func main() {
    // Arrays (fixed size)
    var numbers [5]int = [5]int{1, 2, 3, 4, 5}
    fmt.Println("Array:", numbers)
    fmt.Println("Length:", len(numbers))
    fmt.Println("First element:", numbers[0])
    
    // Array initialization
    fruits := [3]string{"apple", "banana", "orange"}
    fmt.Println("Fruits:", fruits)
    
    // Slices (dynamic arrays)
    var scores []int
    scores = append(scores, 85, 92, 78, 96)
    fmt.Println("Scores:", scores)
    fmt.Println("Length:", len(scores))
    fmt.Println("Capacity:", cap(scores))
    
    // Slice initialization
    colors := []string{"red", "green", "blue"}
    fmt.Println("Colors:", colors)
    
    // Slice operations
    numbers2 := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    fmt.Println("Original:", numbers2)
    fmt.Println("First 3:", numbers2[:3])
    fmt.Println("Last 3:", numbers2[len(numbers2)-3:])
    fmt.Println("Middle 3:", numbers2[3:6])
    
    // Iterating over slices
    for i, color := range colors {
        fmt.Printf("Index %d: %s\n", i, color)
    }
}
```

### Maps

```go
package main

import "fmt"

func main() {
    // Map declaration and initialization
    ages := make(map[string]int)
    ages["Alice"] = 25
    ages["Bob"] = 30
    ages["Charlie"] = 35
    
    fmt.Println("Ages:", ages)
    fmt.Println("Alice's age:", ages["Alice"])
    
    // Map with initial values
    capitals := map[string]string{
        "USA": "Washington D.C.",
        "UK":  "London",
        "France": "Paris",
    }
    fmt.Println("Capitals:", capitals)
    
    // Check if key exists
    if age, exists := ages["David"]; exists {
        fmt.Println("David's age:", age)
    } else {
        fmt.Println("David not found")
    }
    
    // Delete from map
    delete(ages, "Charlie")
    fmt.Println("After deletion:", ages)
    
    // Iterate over map
    for name, age := range ages {
        fmt.Printf("%s is %d years old\n", name, age)
    }
}
```

---

## 🔄 Control Structures

### If-Else Statements

```go
package main

import "fmt"

func main() {
    age := 20
    
    // Basic if statement
    if age >= 18 {
        fmt.Println("You are an adult")
    }
    
    // If-else statement
    if age >= 18 {
        fmt.Println("You can vote")
    } else {
        fmt.Println("You cannot vote")
    }
    
    // If-else if-else statement
    score := 85
    if score >= 90 {
        fmt.Println("Grade: A")
    } else if score >= 80 {
        fmt.Println("Grade: B")
    } else if score >= 70 {
        fmt.Println("Grade: C")
    } else {
        fmt.Println("Grade: F")
    }
    
    // If with initialization
    if num := 42; num%2 == 0 {
        fmt.Println("Even number")
    } else {
        fmt.Println("Odd number")
    }
}
```

### Switch Statements

```go
package main

import "fmt"

func main() {
    day := "Monday"
    
    // Basic switch
    switch day {
    case "Monday":
        fmt.Println("Start of work week")
    case "Friday":
        fmt.Println("TGIF!")
    case "Saturday", "Sunday":
        fmt.Println("Weekend!")
    default:
        fmt.Println("Regular day")
    }
    
    // Switch with expression
    score := 85
    switch {
    case score >= 90:
        fmt.Println("Excellent!")
    case score >= 80:
        fmt.Println("Good!")
    case score >= 70:
        fmt.Println("Average")
    default:
        fmt.Println("Needs improvement")
    }
    
    // Switch with type assertion
    var i interface{} = "hello"
    switch v := i.(type) {
    case int:
        fmt.Println("Integer:", v)
    case string:
        fmt.Println("String:", v)
    case bool:
        fmt.Println("Boolean:", v)
    default:
        fmt.Println("Unknown type")
    }
}
```

### Loops

```go
package main

import "fmt"

func main() {
    // For loop (traditional)
    for i := 0; i < 5; i++ {
        fmt.Printf("Count: %d\n", i)
    }
    
    // While-like loop
    count := 0
    for count < 3 {
        fmt.Printf("While count: %d\n", count)
        count++
    }
    
    // Infinite loop with break
    num := 0
    for {
        if num >= 3 {
            break
        }
        fmt.Printf("Infinite loop: %d\n", num)
        num++
    }
    
    // Range loop with slice
    fruits := []string{"apple", "banana", "orange"}
    for index, fruit := range fruits {
        fmt.Printf("Index %d: %s\n", index, fruit)
    }
    
    // Range loop with map
    colors := map[string]string{
        "red":   "#FF0000",
        "green": "#00FF00",
        "blue":  "#0000FF",
    }
    for color, hex := range colors {
        fmt.Printf("%s: %s\n", color, hex)
    }
    
    // Range loop with string
    text := "Hello"
    for i, char := range text {
        fmt.Printf("Index %d: %c\n", i, char)
    }
}
```

---

## 🔧 Functions

### Basic Functions

```go
package main

import "fmt"

// Function without parameters and return value
func sayHello() {
    fmt.Println("Hello, World!")
}

// Function with parameters
func greet(name string) {
    fmt.Printf("Hello, %s!\n", name)
}

// Function with return value
func add(a, b int) int {
    return a + b
}

// Function with multiple return values
func divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, fmt.Errorf("division by zero")
    }
    return a / b, nil
}

// Function with named return values
func calculate(a, b int) (sum, product int) {
    sum = a + b
    product = a * b
    return // naked return
}

// Variadic function (variable number of arguments)
func sum(numbers ...int) int {
    total := 0
    for _, num := range numbers {
        total += num
    }
    return total
}

func main() {
    // Call functions
    sayHello()
    greet("Alice")
    
    result := add(5, 3)
    fmt.Println("5 + 3 =", result)
    
    quotient, err := divide(10, 2)
    if err != nil {
        fmt.Println("Error:", err)
    } else {
        fmt.Println("10 / 2 =", quotient)
    }
    
    s, p := calculate(4, 5)
    fmt.Printf("Sum: %d, Product: %d\n", s, p)
    
    total := sum(1, 2, 3, 4, 5)
    fmt.Println("Sum of 1,2,3,4,5 =", total)
}
```

### Higher-Order Functions

```go
package main

import "fmt"

// Function that takes another function as parameter
func applyOperation(a, b int, operation func(int, int) int) int {
    return operation(a, b)
}

// Function that returns another function
func createMultiplier(factor int) func(int) int {
    return func(x int) int {
        return x * factor
    }
}

func main() {
    // Define operation functions
    add := func(a, b int) int { return a + b }
    multiply := func(a, b int) int { return a * b }
    
    // Use higher-order function
    result1 := applyOperation(5, 3, add)
    result2 := applyOperation(5, 3, multiply)
    
    fmt.Println("5 + 3 =", result1)
    fmt.Println("5 * 3 =", result2)
    
    // Create and use function factory
    double := createMultiplier(2)
    triple := createMultiplier(3)
    
    fmt.Println("Double 5:", double(5))
    fmt.Println("Triple 5:", triple(5))
}
```

---

## 🏗️ Structs and Methods

### Basic Structs

```go
package main

import "fmt"

// Define a struct
type Person struct {
    Name    string
    Age     int
    Email   string
    Address Address
}

type Address struct {
    Street string
    City   string
    State  string
    Zip    string
}

// Method on struct (value receiver)
func (p Person) GetInfo() string {
    return fmt.Sprintf("Name: %s, Age: %d, Email: %s", p.Name, p.Age, p.Email)
}

// Method on struct (pointer receiver)
func (p *Person) SetAge(age int) {
    p.Age = age
}

// Method on struct (pointer receiver)
func (p *Person) GetFullAddress() string {
    return fmt.Sprintf("%s, %s, %s %s", p.Address.Street, p.Address.City, p.Address.State, p.Address.Zip)
}

func main() {
    // Create struct instance
    person1 := Person{
        Name:  "John Doe",
        Age:   30,
        Email: "john@example.com",
        Address: Address{
            Street: "123 Main St",
            City:   "New York",
            State:  "NY",
            Zip:    "10001",
        },
    }
    
    // Access struct fields
    fmt.Println("Name:", person1.Name)
    fmt.Println("Age:", person1.Age)
    
    // Call methods
    fmt.Println(person1.GetInfo())
    fmt.Println("Address:", person1.GetFullAddress())
    
    // Modify struct
    person1.SetAge(31)
    fmt.Println("Updated age:", person1.Age)
    
    // Create struct with pointer
    person2 := &Person{
        Name:  "Jane Smith",
        Age:   25,
        Email: "jane@example.com",
    }
    
    fmt.Println(person2.GetInfo())
}
```

### Embedded Structs

```go
package main

import "fmt"

// Base struct
type Animal struct {
    Name string
    Age  int
}

func (a Animal) Speak() {
    fmt.Println("Some generic animal sound")
}

// Embedded struct
type Dog struct {
    Animal // Embedded struct
    Breed  string
}

// Override method
func (d Dog) Speak() {
    fmt.Println("Woof! Woof!")
}

// New method
func (d Dog) Fetch() {
    fmt.Println("Fetching the ball!")
}

func main() {
    // Create Dog instance
    dog := Dog{
        Animal: Animal{
            Name: "Buddy",
            Age:  3,
        },
        Breed: "Golden Retriever",
    }
    
    // Access embedded fields
    fmt.Println("Name:", dog.Name)
    fmt.Println("Age:", dog.Age)
    fmt.Println("Breed:", dog.Breed)
    
    // Call methods
    dog.Speak()  // Overridden method
    dog.Fetch()  // New method
}
```

---

## 🔌 Interfaces

### Basic Interfaces

```go
package main

import "fmt"

// Define interface
type Shape interface {
    Area() float64
    Perimeter() float64
}

// Implement interface for Rectangle
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

// Implement interface for Circle
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
    // Create instances
    rect := Rectangle{Width: 5, Height: 3}
    circle := Circle{Radius: 4}
    
    // Use interface
    printShapeInfo(rect)
    printShapeInfo(circle)
    
    // Type assertion
    var shape Shape = rect
    if rect, ok := shape.(Rectangle); ok {
        fmt.Printf("Rectangle width: %.2f, height: %.2f\n", rect.Width, rect.Height)
    }
    
    // Type switch
    shapes := []Shape{rect, circle}
    for i, shape := range shapes {
        switch s := shape.(type) {
        case Rectangle:
            fmt.Printf("Shape %d: Rectangle (%.2f x %.2f)\n", i, s.Width, s.Height)
        case Circle:
            fmt.Printf("Shape %d: Circle (radius: %.2f)\n", i, s.Radius)
        }
    }
}
```

### Empty Interface

```go
package main

import "fmt"

// Empty interface can hold any type
func printValue(i interface{}) {
    switch v := i.(type) {
    case int:
        fmt.Printf("Integer: %d\n", v)
    case string:
        fmt.Printf("String: %s\n", v)
    case bool:
        fmt.Printf("Boolean: %t\n", v)
    default:
        fmt.Printf("Unknown type: %T\n", v)
    }
}

func main() {
    printValue(42)
    printValue("Hello")
    printValue(true)
    printValue(3.14)
}
```

---

## ⚡ Concurrency

### Goroutines

```go
package main

import (
    "fmt"
    "time"
)

// Function to run in goroutine
func sayHello(name string) {
    for i := 0; i < 3; i++ {
        fmt.Printf("Hello %s! (goroutine)\n", name)
        time.Sleep(100 * time.Millisecond)
    }
}

func main() {
    // Start goroutine
    go sayHello("Alice")
    go sayHello("Bob")
    
    // Main goroutine continues
    for i := 0; i < 3; i++ {
        fmt.Printf("Hello from main! %d\n", i)
        time.Sleep(100 * time.Millisecond)
    }
    
    // Wait for goroutines to complete
    time.Sleep(500 * time.Millisecond)
    fmt.Println("Done!")
}
```

### Channels

```go
package main

import (
    "fmt"
    "time"
)

// Function that sends data to channel
func sendData(ch chan string) {
    messages := []string{"Hello", "World", "Go", "Programming"}
    
    for _, msg := range messages {
        ch <- msg // Send data to channel
        time.Sleep(100 * time.Millisecond)
    }
    close(ch) // Close channel when done
}

// Function that receives data from channel
func receiveData(ch chan string) {
    for msg := range ch { // Receive data from channel
        fmt.Printf("Received: %s\n", msg)
    }
}

func main() {
    // Create channel
    ch := make(chan string)
    
    // Start goroutines
    go sendData(ch)
    go receiveData(ch)
    
    // Wait for completion
    time.Sleep(1 * time.Second)
    fmt.Println("Done!")
}
```

### Buffered Channels

```go
package main

import (
    "fmt"
    "time"
)

func main() {
    // Create buffered channel
    ch := make(chan int, 3)
    
    // Send data (won't block until buffer is full)
    ch <- 1
    ch <- 2
    ch <- 3
    
    fmt.Println("Sent 3 values to buffered channel")
    
    // Receive data
    fmt.Println("Received:", <-ch)
    fmt.Println("Received:", <-ch)
    fmt.Println("Received:", <-ch)
    
    // Close channel
    close(ch)
}
```

### Select Statement

```go
package main

import (
    "fmt"
    "time"
)

func main() {
    ch1 := make(chan string)
    ch2 := make(chan string)
    
    // Start goroutines
    go func() {
        time.Sleep(1 * time.Second)
        ch1 <- "Message from channel 1"
    }()
    
    go func() {
        time.Sleep(2 * time.Second)
        ch2 <- "Message from channel 2"
    }()
    
    // Select statement
    for i := 0; i < 2; i++ {
        select {
        case msg1 := <-ch1:
            fmt.Println("Received from ch1:", msg1)
        case msg2 := <-ch2:
            fmt.Println("Received from ch2:", msg2)
        case <-time.After(3 * time.Second):
            fmt.Println("Timeout!")
        }
    }
}
```

---

## ❌ Error Handling

### Basic Error Handling

```go
package main

import (
    "errors"
    "fmt"
    "math"
)

// Function that returns an error
func divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}

// Function that returns a custom error
func sqrt(x float64) (float64, error) {
    if x < 0 {
        return 0, fmt.Errorf("square root of negative number: %f", x)
    }
    return math.Sqrt(x), nil
}

func main() {
    // Handle error
    result, err := divide(10, 2)
    if err != nil {
        fmt.Println("Error:", err)
    } else {
        fmt.Println("Result:", result)
    }
    
    // Handle error with sqrt
    sqrtResult, err := sqrt(-4)
    if err != nil {
        fmt.Println("Error:", err)
    } else {
        fmt.Println("Square root:", sqrtResult)
    }
    
    // Handle error with sqrt (positive number)
    sqrtResult2, err := sqrt(16)
    if err != nil {
        fmt.Println("Error:", err)
    } else {
        fmt.Println("Square root of 16:", sqrtResult2)
    }
}
```

### Custom Error Types

```go
package main

import (
    "fmt"
    "time"
)

// Custom error type
type ValidationError struct {
    Field   string
    Message string
    Time    time.Time
}

// Implement error interface
func (e ValidationError) Error() string {
    return fmt.Sprintf("validation error in field '%s': %s (at %s)", 
        e.Field, e.Message, e.Time.Format("2006-01-02 15:04:05"))
}

// Function that returns custom error
func validateEmail(email string) error {
    if email == "" {
        return ValidationError{
            Field:   "email",
            Message: "email cannot be empty",
            Time:    time.Now(),
        }
    }
    if len(email) < 5 {
        return ValidationError{
            Field:   "email",
            Message: "email must be at least 5 characters",
            Time:    time.Now(),
        }
    }
    return nil
}

func main() {
    // Test validation
    err := validateEmail("")
    if err != nil {
        fmt.Println("Error:", err)
    }
    
    err = validateEmail("ab")
    if err != nil {
        fmt.Println("Error:", err)
    }
    
    err = validateEmail("user@example.com")
    if err != nil {
        fmt.Println("Error:", err)
    } else {
        fmt.Println("Email is valid!")
    }
}
```

---

## 📦 Packages and Modules

### Creating a Module

Create a new directory and initialize a module:

```bash
mkdir myproject
cd myproject
go mod init myproject
```

### Package Structure

```
myproject/
├── go.mod
├── main.go
├── math/
│   └── operations.go
└── utils/
    └── helpers.go
```

### math/operations.go

```go
package math

// Add adds two integers
func Add(a, b int) int {
    return a + b
}

// Subtract subtracts b from a
func Subtract(a, b int) int {
    return a - b
}

// Multiply multiplies two integers
func Multiply(a, b int) int {
    return a * b
}

// Divide divides a by b
func Divide(a, b int) (int, error) {
    if b == 0 {
        return 0, fmt.Errorf("division by zero")
    }
    return a / b, nil
}
```

### utils/helpers.go

```go
package utils

import "fmt"

// PrintMessage prints a message
func PrintMessage(msg string) {
    fmt.Println("Message:", msg)
}

// IsEven checks if a number is even
func IsEven(n int) bool {
    return n%2 == 0
}

// Max returns the maximum of two integers
func Max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

### main.go

```go
package main

import (
    "fmt"
    "myproject/math"
    "myproject/utils"
)

func main() {
    // Use math package
    sum := math.Add(5, 3)
    fmt.Println("5 + 3 =", sum)
    
    diff := math.Subtract(10, 4)
    fmt.Println("10 - 4 =", diff)
    
    product := math.Multiply(6, 7)
    fmt.Println("6 * 7 =", product)
    
    quotient, err := math.Divide(15, 3)
    if err != nil {
        fmt.Println("Error:", err)
    } else {
        fmt.Println("15 / 3 =", quotient)
    }
    
    // Use utils package
    utils.PrintMessage("Hello from utils!")
    fmt.Println("Is 8 even?", utils.IsEven(8))
    fmt.Println("Max of 10 and 20:", utils.Max(10, 20))
}
```

---

## 🚀 Hands-on Projects

### Project 1: Calculator

Create a simple calculator program:

```go
package main

import (
    "bufio"
    "fmt"
    "os"
    "strconv"
    "strings"
)

func main() {
    reader := bufio.NewReader(os.Stdin)
    
    for {
        fmt.Print("Enter operation (+, -, *, /) or 'quit' to exit: ")
        operation, _ := reader.ReadString('\n')
        operation = strings.TrimSpace(operation)
        
        if operation == "quit" {
            break
        }
        
        fmt.Print("Enter first number: ")
        num1Str, _ := reader.ReadString('\n')
        num1, err := strconv.ParseFloat(strings.TrimSpace(num1Str), 64)
        if err != nil {
            fmt.Println("Invalid number!")
            continue
        }
        
        fmt.Print("Enter second number: ")
        num2Str, _ := reader.ReadString('\n')
        num2, err := strconv.ParseFloat(strings.TrimSpace(num2Str), 64)
        if err != nil {
            fmt.Println("Invalid number!")
            continue
        }
        
        var result float64
        switch operation {
        case "+":
            result = num1 + num2
        case "-":
            result = num1 - num2
        case "*":
            result = num1 * num2
        case "/":
            if num2 == 0 {
                fmt.Println("Division by zero!")
                continue
            }
            result = num1 / num2
        default:
            fmt.Println("Invalid operation!")
            continue
        }
        
        fmt.Printf("Result: %.2f\n\n", result)
    }
}
```

### Project 2: Todo List

Create a simple todo list application:

```go
package main

import (
    "bufio"
    "fmt"
    "os"
    "strings"
)

type Todo struct {
    ID          int
    Description string
    Completed   bool
}

type TodoList struct {
    todos []Todo
    nextID int
}

func (tl *TodoList) Add(description string) {
    todo := Todo{
        ID:          tl.nextID,
        Description: description,
        Completed:   false,
    }
    tl.todos = append(tl.todos, todo)
    tl.nextID++
    fmt.Printf("Added todo: %s\n", description)
}

func (tl *TodoList) List() {
    if len(tl.todos) == 0 {
        fmt.Println("No todos found!")
        return
    }
    
    fmt.Println("\nTodo List:")
    for _, todo := range tl.todos {
        status := " "
        if todo.Completed {
            status = "✓"
        }
        fmt.Printf("%d. [%s] %s\n", todo.ID, status, todo.Description)
    }
}

func (tl *TodoList) Complete(id int) {
    for i, todo := range tl.todos {
        if todo.ID == id {
            tl.todos[i].Completed = true
            fmt.Printf("Completed todo: %s\n", todo.Description)
            return
        }
    }
    fmt.Println("Todo not found!")
}

func (tl *TodoList) Delete(id int) {
    for i, todo := range tl.todos {
        if todo.ID == id {
            tl.todos = append(tl.todos[:i], tl.todos[i+1:]...)
            fmt.Printf("Deleted todo: %s\n", todo.Description)
            return
        }
    }
    fmt.Println("Todo not found!")
}

func main() {
    todoList := &TodoList{nextID: 1}
    reader := bufio.NewReader(os.Stdin)
    
    for {
        fmt.Print("\nTodo List Manager\n")
        fmt.Println("1. Add todo")
        fmt.Println("2. List todos")
        fmt.Println("3. Complete todo")
        fmt.Println("4. Delete todo")
        fmt.Println("5. Exit")
        fmt.Print("Choose an option: ")
        
        choice, _ := reader.ReadString('\n')
        choice = strings.TrimSpace(choice)
        
        switch choice {
        case "1":
            fmt.Print("Enter todo description: ")
            description, _ := reader.ReadString('\n')
            description = strings.TrimSpace(description)
            todoList.Add(description)
            
        case "2":
            todoList.List()
            
        case "3":
            fmt.Print("Enter todo ID to complete: ")
            idStr, _ := reader.ReadString('\n')
            idStr = strings.TrimSpace(idStr)
            var id int
            if _, err := fmt.Sscanf(idStr, "%d", &id); err != nil {
                fmt.Println("Invalid ID!")
                continue
            }
            todoList.Complete(id)
            
        case "4":
            fmt.Print("Enter todo ID to delete: ")
            idStr, _ := reader.ReadString('\n')
            idStr = strings.TrimSpace(idStr)
            var id int
            if _, err := fmt.Sscanf(idStr, "%d", &id); err != nil {
                fmt.Println("Invalid ID!")
                continue
            }
            todoList.Delete(id)
            
        case "5":
            fmt.Println("Goodbye!")
            return
            
        default:
            fmt.Println("Invalid option!")
        }
    }
}
```

---

## 🏋️ Practice Exercises

### Exercise 1: Basic Operations

```go
// Write a function that takes two integers and returns their sum, difference, product, and quotient
func calculate(a, b int) (int, int, int, float64, error) {
    // Your code here
}
```

### Exercise 2: String Manipulation

```go
// Write a function that counts the number of vowels in a string
func countVowels(s string) int {
    // Your code here
}
```

### Exercise 3: Array Operations

```go
// Write a function that finds the maximum and minimum values in a slice of integers
func findMinMax(numbers []int) (int, int) {
    // Your code here
}
```

### Exercise 4: Map Operations

```go
// Write a function that counts the frequency of each word in a string
func wordFrequency(text string) map[string]int {
    // Your code here
}
```

### Exercise 5: Struct and Methods

```go
// Create a BankAccount struct with methods to deposit, withdraw, and check balance
type BankAccount struct {
    // Your code here
}

func (ba *BankAccount) Deposit(amount float64) {
    // Your code here
}

func (ba *BankAccount) Withdraw(amount float64) error {
    // Your code here
}

func (ba *BankAccount) GetBalance() float64 {
    // Your code here
}
```

---

## 🎯 Next Steps

1. **Practice**: Work through the exercises and projects
2. **Read Documentation**: Explore the official Go documentation
3. **Build Projects**: Create your own applications
4. **Learn Advanced Topics**: Study concurrency patterns, testing, and web development
5. **Join Community**: Participate in Go forums and communities

---

**🎉 Congratulations! You now have a solid foundation in Go programming! 🚀**

**Keep practicing and building projects to master Go!**
