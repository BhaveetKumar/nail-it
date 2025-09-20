# 🚀 Go Expertise: From Scratch to Super Duper Pro

> **Complete guide to mastering Go (Golang) from fundamentals to advanced concepts with real-world examples and FAANG interview questions**

## 📋 Table of Contents

1. [Go Fundamentals](#go-fundamentals)
2. [Advanced Go Concepts](#advanced-go-concepts)
3. [Design Patterns in Go](#design-patterns-in-go)
4. [Go Architecture & Best Practices](#go-architecture--best-practices)
5. [Debugging & Error Handling](#debugging--error-handling)
6. [Performance Optimization](#performance-optimization)
7. [Concurrency Deep Dive](#concurrency-deep-dive)
8. [FAANG Interview Questions](#faang-interview-questions)

---

## 🎯 Go Fundamentals

### **1. Go Basics & Syntax**

#### **Variables and Constants**

```go
package main

import "fmt"

func main() {
    // Variable declarations - explicit type declaration
    // var keyword followed by variable name and type
    var name string = "Go"        // String variable with explicit initialization
    var age int = 10              // Integer variable
    var isAwesome bool = true     // Boolean variable

    // Short declaration operator (:=) - Go infers the type
    // This is the preferred way in Go for local variables
    city := "San Francisco"       // Type inferred as string
    count := 42                   // Type inferred as int
    flag := true                  // Type inferred as bool

    // Multiple declarations - useful for grouping related variables
    // Parentheses allow multiple variable declarations in one statement
    var (
        firstName = "John"        // All variables in this block are strings
        lastName  = "Doe"         // Go infers type from the first assignment
        email     = "john@example.com"
    )

    // Constants - values that cannot be changed after declaration
    // const keyword followed by name and value
    const pi = 3.14159            // Numeric constant
    const greeting = "Hello"      // String constant
    
    // Multiple constants - grouped for better organization
    const (
        StatusOK    = 200         // HTTP status codes
        StatusError = 500         // Grouped by purpose
    )

    // Printf with format specifiers
    // %s for strings, %d for integers, %v for any value
    fmt.Printf("Name: %s, Age: %d, City: %s\n", name, age, city)
}
```

**Key Concepts Explained:**
- **`var` keyword**: Explicit variable declaration with type
- **`:=` operator**: Short declaration, Go infers type automatically
- **Multiple declarations**: Group related variables using parentheses
- **`const` keyword**: Immutable values, must be known at compile time
- **Type inference**: Go automatically determines type from value
- **Format specifiers**: `%s` (string), `%d` (integer), `%v` (any value)

#### **Data Types**

```go
package main

import "fmt"

func main() {
    // Basic types - Go's primitive data types
    var i int = 42              // Integer: 32 or 64 bits depending on platform
    var f float64 = 3.14        // Floating point: 64-bit precision
    var s string = "Hello, Go!" // String: immutable sequence of bytes
    var b bool = true           // Boolean: true or false

    // Complex types - composite data structures
    
    // Arrays: fixed-size sequence of elements of the same type
    // [5]int means array of 5 integers, size is part of the type
    var arr [5]int = [5]int{1, 2, 3, 4, 5}
    
    // Slices: dynamic arrays, more commonly used than arrays
    // []int means slice of integers, size can change
    var slice []int = []int{1, 2, 3, 4, 5}
    
    // Maps: key-value pairs, similar to hash tables or dictionaries
    // map[string]int means map with string keys and integer values
    var m map[string]int = map[string]int{"a": 1, "b": 2}

    // Pointers: store memory address of a value
    // &i gets the address of variable i
    // *int means pointer to an integer
    var ptr *int = &i
    fmt.Printf("Value: %d, Pointer: %p\n", *ptr, ptr) // *ptr dereferences the pointer

    // Structs: custom types that group related data
    // Define a new type called Person
    type Person struct {
        Name string  // Field name with type
        Age  int     // Another field
    }

    // Create an instance of Person struct
    // Field names can be specified for clarity
    person := Person{Name: "Alice", Age: 30}
    fmt.Printf("Person: %+v\n", person) // %+v shows field names
}
```

**Key Concepts Explained:**
- **Basic types**: `int`, `float64`, `string`, `bool` - Go's primitive types
- **Arrays**: Fixed-size collections `[size]type`, size is part of type
- **Slices**: Dynamic arrays `[]type`, most commonly used collection type
- **Maps**: Key-value pairs `map[keyType]valueType`, like hash tables
- **Pointers**: Store memory addresses, `&` gets address, `*` dereferences
- **Structs**: Custom types grouping related fields, similar to classes
- **Type system**: Go is statically typed, types must be known at compile time

### **2. Functions & Methods**

#### **Function Basics**

```go
package main

import "fmt"

// Basic function - func keyword, name, parameters, return type
// func add(a, b int) int means: function named add takes two int parameters, returns int
func add(a, b int) int {
    return a + b  // return statement with expression
}

// Multiple return values - Go's unique feature for error handling
// Returns both result and error, following Go's error handling convention
func divide(a, b int) (int, error) {
    if b == 0 {
        // fmt.Errorf creates a formatted error message
        return 0, fmt.Errorf("division by zero")
    }
    return a / b, nil  // nil means no error
}

// Named return values - return variables are declared in function signature
// This allows "naked return" - just return without specifying values
func calculate(a, b int) (sum, product int) {
    sum = a + b        // Assign to named return variable
    product = a * b    // Assign to named return variable
    return             // Naked return - returns sum and product
}

// Variadic functions - can accept variable number of arguments
// ...int means zero or more integers
func sum(numbers ...int) int {
    total := 0
    // range iterates over the slice of numbers
    // _ ignores the index, num is the value
    for _, num := range numbers {
        total += num
    }
    return total
}

// Function as parameter - functions are first-class citizens in Go
// op is a function that takes two ints and returns an int
func applyOperation(a, b int, op func(int, int) int) int {
    return op(a, b)  // Call the passed function
}

func main() {
    // Call basic function
    result := add(5, 3)
    fmt.Printf("Add: %d\n", result)

    // Handle multiple return values
    quotient, err := divide(10, 2)
    if err != nil {  // Check for error first (Go idiom)
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Divide: %d\n", quotient)
    }

    // Named return values
    s, p := calculate(4, 5)
    fmt.Printf("Sum: %d, Product: %d\n", s, p)

    // Variadic function call - can pass any number of arguments
    total := sum(1, 2, 3, 4, 5)
    fmt.Printf("Sum of variadic: %d\n", total)

    // Anonymous function (lambda) - function without a name
    multiply := func(x, y int) int { return x * y }
    result = applyOperation(3, 4, multiply)
    fmt.Printf("Apply operation: %d\n", result)
}
```

**Key Concepts Explained:**
- **Function signature**: `func name(params) returnType` - Go's function syntax
- **Multiple returns**: Go's idiomatic way to handle errors and results
- **Named returns**: Variables declared in function signature, enable naked returns
- **Variadic functions**: `...type` accepts variable number of arguments
- **First-class functions**: Functions can be passed as parameters and returned
- **Anonymous functions**: Functions without names, created inline
- **Error handling**: Go's convention of returning `(result, error)` pairs

#### **Methods**

```go
package main

import "fmt"

// Define a struct type
type Rectangle struct {
    Width  float64  // Field with type
    Height float64  // Another field
}

// Value receiver method - receives a copy of the struct
// (r Rectangle) is the receiver - r is the name, Rectangle is the type
// This method cannot modify the original struct
func (r Rectangle) Area() float64 {
    return r.Width * r.Height  // Access fields using dot notation
}

// Pointer receiver method - receives a pointer to the struct
// (r *Rectangle) means r is a pointer to Rectangle
// This method can modify the original struct
func (r *Rectangle) Scale(factor float64) {
    r.Width *= factor   // Modify the original struct through pointer
    r.Height *= factor  // Changes persist after method call
}

// Method on non-struct type - Go allows methods on any type
// MyInt is a custom type based on int
type MyInt int

// Method on custom type - (m MyInt) is the receiver
func (m MyInt) IsEven() bool {
    return m%2 == 0  // Use modulo operator to check if even
}

func main() {
    // Create a Rectangle instance
    rect := Rectangle{Width: 10, Height: 5}
    
    // Call value receiver method - cannot modify rect
    fmt.Printf("Area: %.2f\n", rect.Area())

    // Call pointer receiver method - can modify rect
    rect.Scale(2)  // Go automatically converts &rect to *Rectangle
    fmt.Printf("Scaled area: %.2f\n", rect.Area())

    // Create custom type instance
    num := MyInt(4)  // Type conversion from int to MyInt
    fmt.Printf("Is even: %t\n", num.IsEven())
}
```

**Key Concepts Explained:**
- **Methods**: Functions with a receiver - they belong to a type
- **Value receiver**: `(r Rectangle)` - receives a copy, cannot modify original
- **Pointer receiver**: `(r *Rectangle)` - receives pointer, can modify original
- **Receiver syntax**: `func (receiverName Type) methodName() returnType`
- **Method calls**: `instance.methodName()` - Go handles pointer conversion automatically
- **Custom types**: Can define methods on any type, not just structs
- **Encapsulation**: Methods provide a way to associate behavior with data

### **3. Interfaces**

#### **Interface Basics**

```go
package main

import "fmt"

// Interface definition - defines a set of method signatures
// Any type that implements these methods automatically satisfies the interface
type Shape interface {
    Area() float64      // Method signature - no implementation
    Perimeter() float64 // Another method signature
}

// Circle struct implements Shape interface
type Circle struct {
    Radius float64  // Field to store radius
}

// Implement Area method for Circle - satisfies Shape interface
func (c Circle) Area() float64 {
    return 3.14159 * c.Radius * c.Radius  // π * r²
}

// Implement Perimeter method for Circle - satisfies Shape interface
func (c Circle) Perimeter() float64 {
    return 2 * 3.14159 * c.Radius  // 2 * π * r
}

// Square struct also implements Shape interface
type Square struct {
    Side float64  // Field to store side length
}

// Implement Area method for Square
func (s Square) Area() float64 {
    return s.Side * s.Side  // side²
}

// Implement Perimeter method for Square
func (s Square) Perimeter() float64 {
    return 4 * s.Side  // 4 * side
}

// Function that accepts any Shape - demonstrates polymorphism
// s Shape means s can be any type that implements Shape interface
func printShapeInfo(s Shape) {
    fmt.Printf("Area: %.2f, Perimeter: %.2f\n", s.Area(), s.Perimeter())
}

func main() {
    // Create instances of concrete types
    circle := Circle{Radius: 5}
    square := Square{Side: 4}

    // Both can be passed to printShapeInfo because they implement Shape
    printShapeInfo(circle)  // Circle is implicitly converted to Shape
    printShapeInfo(square)  // Square is implicitly converted to Shape

    // Type assertion - check if interface value is of specific type
    // circle.(Circle) attempts to convert Shape back to Circle
    if c, ok := circle.(Circle); ok {  // ok is true if assertion succeeds
        fmt.Printf("Circle radius: %.2f\n", c.Radius)
    }

    // Type switch - switch on the type of interface value
    shapes := []Shape{circle, square}  // Slice of Shape interface
    for _, shape := range shapes {
        switch s := shape.(type) {  // Type switch syntax
        case Circle:
            fmt.Printf("Circle with radius: %.2f\n", s.Radius)
        case Square:
            fmt.Printf("Square with side: %.2f\n", s.Side)
        }
    }
}
```

**Key Concepts Explained:**
- **Interface**: Defines method signatures that types must implement
- **Implicit implementation**: Types automatically satisfy interfaces if they have the required methods
- **Polymorphism**: Interface values can hold any type that implements the interface
- **Type assertion**: `value.(Type)` - converts interface back to concrete type
- **Type switch**: `switch value.(type)` - switches based on the underlying type
- **Duck typing**: "If it walks like a duck and quacks like a duck, it's a duck"
- **Interface satisfaction**: No explicit declaration needed - Go checks at compile time

---

## 🔥 Advanced Go Concepts

### **4. Goroutines & Channels**

#### **Goroutines**

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

// Worker function that processes jobs from a channel
// id: worker identifier for logging
// jobs: read-only channel (<-chan) that receives jobs
// results: write-only channel (chan<-) that sends results
func worker(id int, jobs <-chan int, results chan<- int) {
    // Range over jobs channel - blocks until channel is closed
    for j := range jobs {
        fmt.Printf("Worker %d processing job %d\n", id, j)
        time.Sleep(time.Second)  // Simulate work
        results <- j * 2         // Send result back
    }
}

func main() {
    // Create buffered channels - can hold 100 items without blocking
    jobs := make(chan int, 100)    // Channel for sending jobs
    results := make(chan int, 100) // Channel for receiving results

    // Start 3 worker goroutines
    // go keyword starts a new goroutine (lightweight thread)
    for w := 1; w <= 3; w++ {
        go worker(w, jobs, results)  // Each worker runs concurrently
    }

    // Send 5 jobs to the jobs channel
    for j := 1; j <= 5; j++ {
        jobs <- j  // Send job to channel
    }
    close(jobs)  // Close channel to signal no more jobs

    // Collect results from all workers
    for a := 1; a <= 5; a++ {
        result := <-results  // Receive result from channel
        fmt.Printf("Result: %d\n", result)
    }
}
```

**Key Concepts Explained:**
- **Goroutines**: Lightweight threads managed by Go runtime, not OS threads
- **`go` keyword**: Starts a new goroutine that runs concurrently
- **Channels**: Communication mechanism between goroutines
- **Channel directions**: `<-chan` (receive only), `chan<-` (send only), `chan` (bidirectional)
- **Buffered channels**: `make(chan int, 100)` - can hold 100 items without blocking
- **Channel operations**: `<-` for sending/receiving, `close()` to signal completion
- **Range over channels**: `for item := range channel` - receives until channel is closed
- **Concurrency**: Multiple goroutines run concurrently, not necessarily in parallel

#### **Channels**

```go
package main

import (
    "fmt"
    "time"
)

func main() {
    // Unbuffered channel - synchronous communication
    // Sender blocks until receiver is ready, and vice versa
    ch := make(chan string)

    // Send in goroutine - prevents deadlock
    go func() {
        ch <- "Hello from goroutine!"  // Send message to channel
    }()

    // Receive - blocks until value is available
    msg := <-ch
    fmt.Println(msg)

    // Buffered channel - asynchronous communication
    // Can hold up to 2 values without blocking
    buffered := make(chan int, 2)
    buffered <- 1  // Send first value
    buffered <- 2  // Send second value
    // buffered <- 3 // This would block - buffer is full

    fmt.Println(<-buffered)  // Receive first value
    fmt.Println(<-buffered)  // Receive second value

    // Channel directions - demonstrate send-only and receive-only channels
    go sendOnly(buffered)    // Function that can only send
    go receiveOnly(buffered) // Function that can only receive

    time.Sleep(time.Second)  // Wait for goroutines to complete
}

// Function with send-only channel parameter
// chan<- int means this function can only send to the channel
func sendOnly(ch chan<- int) {
    ch <- 42  // Send value to channel
}

// Function with receive-only channel parameter
// <-chan int means this function can only receive from the channel
func receiveOnly(ch <-chan int) {
    value := <-ch  // Receive value from channel
    fmt.Printf("Received: %d\n", value)
}
```

**Key Concepts Explained:**
- **Unbuffered channels**: Synchronous communication, sender and receiver must be ready
- **Buffered channels**: Asynchronous communication, can hold multiple values
- **Channel directions**: `chan<-` (send only), `<-chan` (receive only), `chan` (bidirectional)
- **Deadlock prevention**: Use goroutines to avoid blocking on unbuffered channels
- **Channel capacity**: Buffered channels have a capacity limit
- **Function parameters**: Channel directions can be specified in function signatures

### **5. Select Statement**

```go
package main

import (
    "fmt"
    "time"
)

func main() {
    // Create two channels for demonstration
    ch1 := make(chan string)
    ch2 := make(chan string)

    // Start first goroutine - sends message after 1 second
    go func() {
        time.Sleep(1 * time.Second)
        ch1 <- "from ch1"  // Send message to ch1
    }()

    // Start second goroutine - sends message after 2 seconds
    go func() {
        time.Sleep(2 * time.Second)
        ch2 <- "from ch2"  // Send message to ch2
    }()

    // Select statement - waits for any case to be ready
    // This loop will run twice to receive both messages
    for i := 0; i < 2; i++ {
        select {
        case msg1 := <-ch1:  // Case for ch1 channel
            fmt.Println(msg1)
        case msg2 := <-ch2:  // Case for ch2 channel
            fmt.Println(msg2)
        case <-time.After(3 * time.Second):  // Timeout case
            fmt.Println("timeout")
        }
    }
}
```

**Key Concepts Explained:**
- **Select statement**: Non-blocking channel operations, waits for any case to be ready
- **Multiple channels**: Can handle multiple channels simultaneously
- **Timeout handling**: `time.After()` provides timeout functionality
- **Non-blocking**: Select doesn't block if no case is ready (unless all cases are blocking)
- **Random selection**: If multiple cases are ready, one is chosen randomly
- **Default case**: Can add `default:` for non-blocking behavior

### **6. Context Package**

```go
package main

import (
    "context"
    "fmt"
    "time"
)

// Function that respects context cancellation
// ctx context.Context is the first parameter (Go convention)
func longRunningTask(ctx context.Context) error {
    for {
        select {
        case <-ctx.Done():  // Check if context is cancelled
            return ctx.Err()  // Return the error (timeout or cancellation)
        default:
            fmt.Println("Working...")
            time.Sleep(500 * time.Millisecond)  // Simulate work
        }
    }
}

func main() {
    // Create context with timeout - automatically cancels after 2 seconds
    // context.Background() is the root context
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()  // Always call cancel to free resources

    // Start long-running task in goroutine
    go func() {
        if err := longRunningTask(ctx); err != nil {
            fmt.Printf("Task failed: %v\n", err)
        }
    }()

    // Wait for 3 seconds to see the timeout in action
    time.Sleep(3 * time.Second)
}
```

**Key Concepts Explained:**
- **Context**: Carries deadlines, cancellation signals, and request-scoped values
- **Context.Background()**: Root context, never cancelled
- **WithTimeout**: Creates context that cancels after specified duration
- **ctx.Done()**: Channel that closes when context is cancelled
- **ctx.Err()**: Returns the error that caused cancellation
- **defer cancel()**: Always call cancel to free resources
- **Context propagation**: Pass context through function calls to enable cancellation

---

## 🎨 Design Patterns in Go

### **7. Singleton Pattern**

```go
package main

import (
    "fmt"
    "sync"
)

// Singleton struct - only one instance should exist
type Singleton struct {
    data string  // Some data to demonstrate the pattern
}

// Package-level variables for singleton implementation
var (
    instance *Singleton  // Pointer to the single instance
    once     sync.Once   // Ensures initialization happens only once
)

// GetInstance returns the singleton instance
// sync.Once ensures the initialization function runs only once
func GetInstance() *Singleton {
    once.Do(func() {
        // This function will only execute once, even if called multiple times
        instance = &Singleton{data: "initialized"}
    })
    return instance
}

func main() {
    // Get two references to the singleton
    s1 := GetInstance()
    s2 := GetInstance()

    // Both should be the same instance
    fmt.Printf("s1 == s2: %t\n", s1 == s2)  // Should print true
    fmt.Printf("s1.data: %s\n", s1.data)    // Access the data
}
```

**Key Concepts Explained:**
- **Singleton pattern**: Ensures only one instance of a class exists
- **sync.Once**: Go's mechanism to ensure a function runs only once
- **Thread-safe**: sync.Once is safe for concurrent access
- **Lazy initialization**: Instance is created only when first requested
- **Global access**: Provides a global point of access to the instance
- **Memory efficiency**: Only one instance exists in memory

### **8. Factory Pattern**

```go
package main

import "fmt"

// Animal interface defines the contract for all animals
type Animal interface {
    Speak() string  // Method that all animals must implement
}

// Dog struct implements Animal interface
type Dog struct{}
func (d Dog) Speak() string { return "Woof!" }

// Cat struct implements Animal interface
type Cat struct{}
func (c Cat) Speak() string { return "Meow!" }

// AnimalFactory creates animals based on type
type AnimalFactory struct{}

// CreateAnimal is the factory method that creates animals
// Returns Animal interface, not concrete type
func (af AnimalFactory) CreateAnimal(animalType string) Animal {
    switch animalType {
    case "dog":
        return Dog{}  // Return Dog instance
    case "cat":
        return Cat{}  // Return Cat instance
    default:
        return nil    // Return nil for unknown types
    }
}

func main() {
    // Create factory instance
    factory := AnimalFactory{}

    // Use factory to create animals
    dog := factory.CreateAnimal("dog")  // Returns Dog instance as Animal interface
    cat := factory.CreateAnimal("cat")  // Returns Cat instance as Animal interface

    // Call methods through interface
    fmt.Println(dog.Speak())  // Calls Dog.Speak()
    fmt.Println(cat.Speak())  // Calls Cat.Speak()
}
```

**Key Concepts Explained:**
- **Factory pattern**: Creates objects without specifying their exact class
- **Interface-based**: Returns interface type, not concrete type
- **Encapsulation**: Hides object creation logic from client code
- **Polymorphism**: Client code works with interface, not concrete types
- **Switch statement**: Determines which concrete type to create
- **Interface implementation**: Both Dog and Cat implement Animal interface

### **9. Observer Pattern**

```go
package main

import "fmt"

// Observer interface defines the contract for observers
type Observer interface {
    Update(message string)  // Method called when subject changes
}

// Subject struct maintains a list of observers and notifies them
type Subject struct {
    observers []Observer  // Slice of observers
}

// AddObserver adds a new observer to the list
func (s *Subject) AddObserver(o Observer) {
    s.observers = append(s.observers, o)  // Append observer to slice
}

// NotifyObservers notifies all observers of a change
func (s *Subject) NotifyObservers(message string) {
    // Iterate through all observers and call their Update method
    for _, observer := range s.observers {
        observer.Update(message)  // Call Update method on each observer
    }
}

// ConcreteObserver implements the Observer interface
type ConcreteObserver struct {
    name string  // Observer identifier
}

// Update method implements the Observer interface
func (co ConcreteObserver) Update(message string) {
    fmt.Printf("%s received: %s\n", co.name, message)
}

func main() {
    // Create subject
    subject := &Subject{}

    // Create observers
    observer1 := ConcreteObserver{name: "Observer1"}
    observer2 := ConcreteObserver{name: "Observer2"}

    // Register observers with subject
    subject.AddObserver(observer1)
    subject.AddObserver(observer2)

    // Notify all observers
    subject.NotifyObservers("Hello, observers!")
}
```

**Key Concepts Explained:**
- **Observer pattern**: Defines a one-to-many dependency between objects
- **Subject**: Maintains a list of observers and notifies them of changes
- **Observer**: Interface that observers must implement
- **Loose coupling**: Subject doesn't know about concrete observer types
- **Event-driven**: Observers are notified when subject state changes
- **Dynamic relationships**: Observers can be added/removed at runtime

---

## 🏗️ Go Architecture & Best Practices

### **10. Project Structure**

```
my-go-project/
├── cmd/
│   └── server/
│       └── main.go
├── internal/
│   ├── config/
│   ├── handlers/
│   ├── models/
│   └── services/
├── pkg/
│   └── utils/
├── api/
│   └── openapi.yaml
├── migrations/
├── tests/
├── go.mod
├── go.sum
├── Dockerfile
└── README.md
```

### **11. Error Handling Best Practices**

```go
package main

import (
    "errors"
    "fmt"
    "log"
)

// Custom error types
type ValidationError struct {
    Field   string
    Message string
}

func (e ValidationError) Error() string {
    return fmt.Sprintf("validation error on field %s: %s", e.Field, e.Message)
}

type BusinessError struct {
    Code    int
    Message string
}

func (e BusinessError) Error() string {
    return fmt.Sprintf("business error %d: %s", e.Code, e.Message)
}

// Error wrapping
func processUser(userID string) error {
    if userID == "" {
        return fmt.Errorf("processUser: %w", ValidationError{
            Field:   "userID",
            Message: "cannot be empty",
        })
    }

    if userID == "invalid" {
        return fmt.Errorf("processUser: %w", BusinessError{
            Code:    1001,
            Message: "user not found",
        })
    }

    return nil
}

// Error handling with context
func handleUserRequest(userID string) error {
    if err := processUser(userID); err != nil {
        // Log the error with context
        log.Printf("Failed to process user %s: %v", userID, err)

        // Check error type
        var validationErr ValidationError
        if errors.As(err, &validationErr) {
            return fmt.Errorf("invalid request: %w", err)
        }

        var businessErr BusinessError
        if errors.As(err, &businessErr) {
            return fmt.Errorf("business logic error: %w", err)
        }

        return fmt.Errorf("unexpected error: %w", err)
    }

    return nil
}

func main() {
    if err := handleUserRequest(""); err != nil {
        fmt.Printf("Error: %v\n", err)
    }

    if err := handleUserRequest("invalid"); err != nil {
        fmt.Printf("Error: %v\n", err)
    }
}
```

### **12. Configuration Management**

```go
package main

import (
    "encoding/json"
    "flag"
    "fmt"
    "os"
)

type Config struct {
    Server   ServerConfig   `json:"server"`
    Database DatabaseConfig `json:"database"`
    Redis    RedisConfig    `json:"redis"`
}

type ServerConfig struct {
    Host string `json:"host"`
    Port int    `json:"port"`
}

type DatabaseConfig struct {
    Host     string `json:"host"`
    Port     int    `json:"port"`
    Username string `json:"username"`
    Password string `json:"password"`
    Database string `json:"database"`
}

type RedisConfig struct {
    Host string `json:"host"`
    Port int    `json:"port"`
}

func LoadConfig() (*Config, error) {
    configFile := flag.String("config", "config.json", "Configuration file path")
    flag.Parse()

    file, err := os.Open(*configFile)
    if err != nil {
        return nil, fmt.Errorf("failed to open config file: %w", err)
    }
    defer file.Close()

    var config Config
    decoder := json.NewDecoder(file)
    if err := decoder.Decode(&config); err != nil {
        return nil, fmt.Errorf("failed to decode config: %w", err)
    }

    return &config, nil
}

func main() {
    config, err := LoadConfig()
    if err != nil {
        fmt.Printf("Failed to load config: %v\n", err)
        return
    }

    fmt.Printf("Server: %s:%d\n", config.Server.Host, config.Server.Port)
    fmt.Printf("Database: %s:%d\n", config.Database.Host, config.Database.Port)
}
```

---

## 🐛 Debugging & Error Handling

### **13. Debugging Techniques**

```go
package main

import (
    "fmt"
    "log"
    "runtime"
    "time"
)

// Debug logging
func debugLog(message string) {
    _, file, line, ok := runtime.Caller(1)
    if ok {
        log.Printf("[DEBUG] %s:%d - %s", file, line, message)
    }
}

// Performance timing
func timeFunction(fn func()) time.Duration {
    start := time.Now()
    fn()
    return time.Since(start)
}

// Memory usage
func printMemUsage() {
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    fmt.Printf("Alloc = %d KB", bToKb(m.Alloc))
    fmt.Printf("\tTotalAlloc = %d KB", bToKb(m.TotalAlloc))
    fmt.Printf("\tSys = %d KB", bToKb(m.Sys))
    fmt.Printf("\tNumGC = %d\n", m.NumGC)
}

func bToKb(b uint64) uint64 {
    return b / 1024
}

func main() {
    debugLog("Starting application")

    duration := timeFunction(func() {
        // Simulate some work
        time.Sleep(100 * time.Millisecond)
    })

    fmt.Printf("Function took: %v\n", duration)
    printMemUsage()
}
```

### **14. Profiling**

```go
package main

import (
    "fmt"
    "log"
    "net/http"
    _ "net/http/pprof"
    "runtime"
    "time"
)

func cpuIntensiveTask() {
    for i := 0; i < 1000000; i++ {
        _ = i * i
    }
}

func memoryIntensiveTask() {
    data := make([]byte, 1024*1024) // 1MB
    for i := range data {
        data[i] = byte(i % 256)
    }
}

func main() {
    // Start pprof server
    go func() {
        log.Println(http.ListenAndServe("localhost:6060", nil))
    }()

    // Simulate some work
    for i := 0; i < 10; i++ {
        cpuIntensiveTask()
        memoryIntensiveTask()
        runtime.GC() // Force garbage collection
        time.Sleep(100 * time.Millisecond)
    }

    fmt.Println("Profiling server running on http://localhost:6060/debug/pprof/")
    fmt.Println("Use 'go tool pprof http://localhost:6060/debug/pprof/profile' for CPU profiling")
    fmt.Println("Use 'go tool pprof http://localhost:6060/debug/pprof/heap' for memory profiling")

    // Keep the program running
    select {}
}
```

---

## ⚡ Performance Optimization

### **15. Memory Optimization**

```go
package main

import (
    "fmt"
    "runtime"
    "sync"
    "time"
)

// Object pooling
type ObjectPool struct {
    pool sync.Pool
}

func NewObjectPool() *ObjectPool {
    return &ObjectPool{
        pool: sync.Pool{
            New: func() interface{} {
                return make([]byte, 1024)
            },
        },
    }
}

func (op *ObjectPool) Get() []byte {
    return op.pool.Get().([]byte)
}

func (op *ObjectPool) Put(obj []byte) {
    op.pool.Put(obj)
}

// String builder optimization
func buildStringOptimized(parts []string) string {
    var builder strings.Builder
    builder.Grow(1000) // Pre-allocate capacity

    for _, part := range parts {
        builder.WriteString(part)
    }

    return builder.String()
}

// Slice pre-allocation
func processDataOptimized(data []int) []int {
    result := make([]int, 0, len(data)) // Pre-allocate capacity

    for _, item := range data {
        if item%2 == 0 {
            result = append(result, item*2)
        }
    }

    return result
}

func main() {
    // Object pooling example
    pool := NewObjectPool()

    obj := pool.Get()
    // Use the object
    fmt.Printf("Got object of length: %d\n", len(obj))
    pool.Put(obj)

    // String building example
    parts := []string{"Hello", " ", "World", "!"}
    result := buildStringOptimized(parts)
    fmt.Printf("Built string: %s\n", result)

    // Slice optimization example
    data := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    processed := processDataOptimized(data)
    fmt.Printf("Processed data: %v\n", processed)
}
```

### **16. Concurrency Optimization**

```go
package main

import (
    "fmt"
    "runtime"
    "sync"
    "time"
)

// Worker pool pattern
type WorkerPool struct {
    workers    int
    jobs       chan func()
    wg         sync.WaitGroup
}

func NewWorkerPool(workers int) *WorkerPool {
    return &WorkerPool{
        workers: workers,
        jobs:    make(chan func(), workers*2),
    }
}

func (wp *WorkerPool) Start() {
    for i := 0; i < wp.workers; i++ {
        wp.wg.Add(1)
        go wp.worker()
    }
}

func (wp *WorkerPool) worker() {
    defer wp.wg.Done()
    for job := range wp.jobs {
        job()
    }
}

func (wp *WorkerPool) Submit(job func()) {
    wp.jobs <- job
}

func (wp *WorkerPool) Stop() {
    close(wp.jobs)
    wp.wg.Wait()
}

// Rate limiting
type RateLimiter struct {
    tokens chan struct{}
    ticker *time.Ticker
}

func NewRateLimiter(rate int, per time.Duration) *RateLimiter {
    rl := &RateLimiter{
        tokens: make(chan struct{}, rate),
        ticker: time.NewTicker(per / time.Duration(rate)),
    }

    go rl.refill()
    return rl
}

func (rl *RateLimiter) refill() {
    for range rl.ticker.C {
        select {
        case rl.tokens <- struct{}{}:
        default:
        }
    }
}

func (rl *RateLimiter) Allow() bool {
    select {
    case <-rl.tokens:
        return true
    default:
        return false
    }
}

func (rl *RateLimiter) Stop() {
    rl.ticker.Stop()
}

func main() {
    // Worker pool example
    pool := NewWorkerPool(runtime.NumCPU())
    pool.Start()

    for i := 0; i < 10; i++ {
        i := i // Capture loop variable
        pool.Submit(func() {
            fmt.Printf("Processing job %d\n", i)
            time.Sleep(100 * time.Millisecond)
        })
    }

    pool.Stop()

    // Rate limiting example
    limiter := NewRateLimiter(5, time.Second)
    defer limiter.Stop()

    for i := 0; i < 10; i++ {
        if limiter.Allow() {
            fmt.Printf("Request %d allowed\n", i)
        } else {
            fmt.Printf("Request %d rate limited\n", i)
        }
        time.Sleep(100 * time.Millisecond)
    }
}
```

---

## 🔄 Concurrency Deep Dive

### **17. Advanced Concurrency Patterns**

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// Fan-out, Fan-in pattern
func fanOut(input <-chan int, workers int) []<-chan int {
    outputs := make([]<-chan int, workers)

    for i := 0; i < workers; i++ {
        output := make(chan int)
        outputs[i] = output

        go func() {
            defer close(output)
            for value := range input {
                // Simulate processing
                time.Sleep(100 * time.Millisecond)
                output <- value * 2
            }
        }()
    }

    return outputs
}

func fanIn(inputs []<-chan int) <-chan int {
    output := make(chan int)
    var wg sync.WaitGroup

    for _, input := range inputs {
        wg.Add(1)
        go func(ch <-chan int) {
            defer wg.Done()
            for value := range ch {
                output <- value
            }
        }(input)
    }

    go func() {
        wg.Wait()
        close(output)
    }()

    return output
}

// Pipeline pattern
func pipeline(input <-chan int) <-chan int {
    output := make(chan int)

    go func() {
        defer close(output)
        for value := range input {
            // Stage 1: Multiply by 2
            value *= 2

            // Stage 2: Add 1
            value += 1

            output <- value
        }
    }()

    return output
}

// Circuit breaker pattern
type CircuitBreaker struct {
    maxFailures int
    timeout     time.Duration
    failures    int
    lastFailure time.Time
    state       string
    mutex       sync.RWMutex
}

func NewCircuitBreaker(maxFailures int, timeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        maxFailures: maxFailures,
        timeout:     timeout,
        state:       "closed",
    }
}

func (cb *CircuitBreaker) Call(fn func() error) error {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()

    if cb.state == "open" {
        if time.Since(cb.lastFailure) > cb.timeout {
            cb.state = "half-open"
        } else {
            return fmt.Errorf("circuit breaker is open")
        }
    }

    err := fn()
    if err != nil {
        cb.failures++
        cb.lastFailure = time.Now()
        if cb.failures >= cb.maxFailures {
            cb.state = "open"
        }
        return err
    }

    cb.failures = 0
    cb.state = "closed"
    return nil
}

func main() {
    // Fan-out, Fan-in example
    input := make(chan int)
    go func() {
        defer close(input)
        for i := 1; i <= 10; i++ {
            input <- i
        }
    }()

    outputs := fanOut(input, 3)
    result := fanIn(outputs)

    for value := range result {
        fmt.Printf("Processed: %d\n", value)
    }

    // Pipeline example
    input2 := make(chan int)
    go func() {
        defer close(input2)
        for i := 1; i <= 5; i++ {
            input2 <- i
        }
    }()

    output := pipeline(input2)
    for value := range output {
        fmt.Printf("Pipeline result: %d\n", value)
    }

    // Circuit breaker example
    cb := NewCircuitBreaker(3, 5*time.Second)

    for i := 0; i < 5; i++ {
        err := cb.Call(func() error {
            if i < 3 {
                return fmt.Errorf("simulated error")
            }
            return nil
        })

        if err != nil {
            fmt.Printf("Call %d failed: %v\n", i, err)
        } else {
            fmt.Printf("Call %d succeeded\n", i)
        }
    }
}
```

---

## 🎯 FAANG Interview Questions

### **Google Interview Questions**

#### **1. Implement a Concurrent Map**

**Question**: "Implement a thread-safe map that can handle concurrent reads and writes efficiently."

**Answer**:

```go
package main

import (
    "fmt"
    "sync"
)

type ConcurrentMap struct {
    mu   sync.RWMutex
    data map[string]interface{}
}

func NewConcurrentMap() *ConcurrentMap {
    return &ConcurrentMap{
        data: make(map[string]interface{}),
    }
}

func (cm *ConcurrentMap) Get(key string) (interface{}, bool) {
    cm.mu.RLock()
    defer cm.mu.RUnlock()
    value, exists := cm.data[key]
    return value, exists
}

func (cm *ConcurrentMap) Set(key string, value interface{}) {
    cm.mu.Lock()
    defer cm.mu.Unlock()
    cm.data[key] = value
}

func (cm *ConcurrentMap) Delete(key string) {
    cm.mu.Lock()
    defer cm.mu.Unlock()
    delete(cm.data, key)
}

func (cm *ConcurrentMap) Size() int {
    cm.mu.RLock()
    defer cm.mu.RUnlock()
    return len(cm.data)
}

func main() {
    cm := NewConcurrentMap()

    // Concurrent writes
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func(i int) {
            defer wg.Done()
            cm.Set(fmt.Sprintf("key%d", i), fmt.Sprintf("value%d", i))
        }(i)
    }

    wg.Wait()

    // Concurrent reads
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func(i int) {
            defer wg.Done()
            if value, exists := cm.Get(fmt.Sprintf("key%d", i)); exists {
                fmt.Printf("Key%d: %v\n", i, value)
            }
        }(i)
    }

    wg.Wait()
    fmt.Printf("Map size: %d\n", cm.Size())
}
```

#### **2. Implement a Rate Limiter**

**Question**: "Design a rate limiter that can handle different rate limits for different users."

**Answer**:

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type RateLimiter struct {
    limiters map[string]*UserLimiter
    mu       sync.RWMutex
}

type UserLimiter struct {
    tokens   int
    lastTime time.Time
    rate     int
    capacity int
    mu       sync.Mutex
}

func NewRateLimiter() *RateLimiter {
    return &RateLimiter{
        limiters: make(map[string]*UserLimiter),
    }
}

func (rl *RateLimiter) GetLimiter(userID string, rate, capacity int) *UserLimiter {
    rl.mu.Lock()
    defer rl.mu.Unlock()

    if limiter, exists := rl.limiters[userID]; exists {
        return limiter
    }

    limiter := &UserLimiter{
        tokens:   capacity,
        lastTime: time.Now(),
        rate:     rate,
        capacity: capacity,
    }

    rl.limiters[userID] = limiter
    return limiter
}

func (ul *UserLimiter) Allow() bool {
    ul.mu.Lock()
    defer ul.mu.Unlock()

    now := time.Now()
    elapsed := now.Sub(ul.lastTime)

    // Add tokens based on elapsed time
    tokensToAdd := int(elapsed.Seconds()) * ul.rate
    ul.tokens = min(ul.capacity, ul.tokens+tokensToAdd)
    ul.lastTime = now

    if ul.tokens > 0 {
        ul.tokens--
        return true
    }

    return false
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

func main() {
    rl := NewRateLimiter()

    // Test rate limiting
    user1 := rl.GetLimiter("user1", 2, 5) // 2 tokens per second, capacity 5
    user2 := rl.GetLimiter("user2", 1, 3) // 1 token per second, capacity 3

    for i := 0; i < 10; i++ {
        if user1.Allow() {
            fmt.Printf("User1 request %d: allowed\n", i)
        } else {
            fmt.Printf("User1 request %d: rate limited\n", i)
        }

        if user2.Allow() {
            fmt.Printf("User2 request %d: allowed\n", i)
        } else {
            fmt.Printf("User2 request %d: rate limited\n", i)
        }

        time.Sleep(500 * time.Millisecond)
    }
}
```

### **Meta Interview Questions**

#### **3. Implement a Message Queue**

**Question**: "Design a message queue system that can handle high throughput and ensure message delivery."

**Answer**:

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Message struct {
    ID        string
    Content   string
    Timestamp time.Time
    Retries   int
}

type MessageQueue struct {
    messages    chan Message
    subscribers map[string][]chan Message
    mu          sync.RWMutex
    wg          sync.WaitGroup
}

func NewMessageQueue(bufferSize int) *MessageQueue {
    return &MessageQueue{
        messages:    make(chan Message, bufferSize),
        subscribers: make(map[string][]chan Message),
    }
}

func (mq *MessageQueue) Subscribe(topic string) <-chan Message {
    mq.mu.Lock()
    defer mq.mu.Unlock()

    ch := make(chan Message, 10)
    mq.subscribers[topic] = append(mq.subscribers[topic], ch)
    return ch
}

func (mq *MessageQueue) Publish(topic string, content string) {
    message := Message{
        ID:        fmt.Sprintf("%d", time.Now().UnixNano()),
        Content:   content,
        Timestamp: time.Now(),
        Retries:   0,
    }

    mq.messages <- message
}

func (mq *MessageQueue) Start() {
    mq.wg.Add(1)
    go mq.processMessages()
}

func (mq *MessageQueue) processMessages() {
    defer mq.wg.Done()

    for message := range mq.messages {
        mq.mu.RLock()
        subscribers := mq.subscribers["default"] // Simplified: all messages go to default topic
        mq.mu.RUnlock()

        for _, ch := range subscribers {
            select {
            case ch <- message:
            default:
                // Subscriber is busy, skip
            }
        }
    }
}

func (mq *MessageQueue) Stop() {
    close(mq.messages)
    mq.wg.Wait()
}

func main() {
    mq := NewMessageQueue(100)
    mq.Start()

    // Subscribe to messages
    ch1 := mq.Subscribe("default")
    ch2 := mq.Subscribe("default")

    // Publish messages
    go func() {
        for i := 0; i < 5; i++ {
            mq.Publish("default", fmt.Sprintf("Message %d", i))
            time.Sleep(100 * time.Millisecond)
        }
    }()

    // Consume messages
    go func() {
        for message := range ch1 {
            fmt.Printf("Consumer 1 received: %s\n", message.Content)
        }
    }()

    go func() {
        for message := range ch2 {
            fmt.Printf("Consumer 2 received: %s\n", message.Content)
        }
    }()

    time.Sleep(2 * time.Second)
    mq.Stop()
}
```

### **Amazon Interview Questions**

#### **4. Implement a Distributed Cache**

**Question**: "Design a distributed cache system with consistent hashing and replication."

**Answer**:

```go
package main

import (
    "crypto/md5"
    "fmt"
    "sort"
    "sync"
    "time"
)

type CacheNode struct {
    ID   string
    Data map[string]interface{}
    mu   sync.RWMutex
}

type ConsistentHash struct {
    nodes    []string
    replicas int
    mu       sync.RWMutex
}

func NewConsistentHash(replicas int) *ConsistentHash {
    return &ConsistentHash{
        replicas: replicas,
    }
}

func (ch *ConsistentHash) AddNode(node string) {
    ch.mu.Lock()
    defer ch.mu.Unlock()

    for i := 0; i < ch.replicas; i++ {
        hash := ch.hash(fmt.Sprintf("%s:%d", node, i))
        ch.nodes = append(ch.nodes, fmt.Sprintf("%d:%s", hash, node))
    }

    sort.Strings(ch.nodes)
}

func (ch *ConsistentHash) GetNode(key string) string {
    ch.mu.RLock()
    defer ch.mu.RUnlock()

    if len(ch.nodes) == 0 {
        return ""
    }

    hash := ch.hash(key)
    idx := sort.Search(len(ch.nodes), func(i int) bool {
        return ch.nodes[i] >= fmt.Sprintf("%d:", hash)
    })

    if idx == len(ch.nodes) {
        idx = 0
    }

    nodeHash := ch.nodes[idx]
    return nodeHash[11:] // Remove hash prefix
}

func (ch *ConsistentHash) hash(key string) uint32 {
    h := md5.Sum([]byte(key))
    return uint32(h[0])<<24 | uint32(h[1])<<16 | uint32(h[2])<<8 | uint32(h[3])
}

type DistributedCache struct {
    nodes  map[string]*CacheNode
    hash   *ConsistentHash
    mu     sync.RWMutex
}

func NewDistributedCache() *DistributedCache {
    return &DistributedCache{
        nodes: make(map[string]*CacheNode),
        hash:  NewConsistentHash(3),
    }
}

func (dc *DistributedCache) AddNode(nodeID string) {
    dc.mu.Lock()
    defer dc.mu.Unlock()

    node := &CacheNode{
        ID:   nodeID,
        Data: make(map[string]interface{}),
    }

    dc.nodes[nodeID] = node
    dc.hash.AddNode(nodeID)
}

func (dc *DistributedCache) Get(key string) (interface{}, bool) {
    dc.mu.RLock()
    nodeID := dc.hash.GetNode(key)
    dc.mu.RUnlock()

    if nodeID == "" {
        return nil, false
    }

    dc.mu.RLock()
    node := dc.nodes[nodeID]
    dc.mu.RUnlock()

    node.mu.RLock()
    defer node.mu.RUnlock()

    value, exists := node.Data[key]
    return value, exists
}

func (dc *DistributedCache) Set(key string, value interface{}) {
    dc.mu.RLock()
    nodeID := dc.hash.GetNode(key)
    dc.mu.RUnlock()

    if nodeID == "" {
        return
    }

    dc.mu.RLock()
    node := dc.nodes[nodeID]
    dc.mu.RUnlock()

    node.mu.Lock()
    defer node.mu.Unlock()

    node.Data[key] = value
}

func main() {
    cache := NewDistributedCache()

    // Add nodes
    cache.AddNode("node1")
    cache.AddNode("node2")
    cache.AddNode("node3")

    // Set values
    cache.Set("key1", "value1")
    cache.Set("key2", "value2")
    cache.Set("key3", "value3")

    // Get values
    if value, exists := cache.Get("key1"); exists {
        fmt.Printf("key1: %v\n", value)
    }

    if value, exists := cache.Get("key2"); exists {
        fmt.Printf("key2: %v\n", value)
    }

    if value, exists := cache.Get("key3"); exists {
        fmt.Printf("key3: %v\n", value)
    }
}
```

---

## 📚 Additional Resources

### **Books**

- [The Go Programming Language](https://www.gopl.io/) - Alan Donovan & Brian Kernighan
- [Effective Go](https://golang.org/doc/effective_go.html/) - Official Go documentation
- [Go in Action](https://www.manning.com/books/go-in-action/) - William Kennedy

### **Online Resources**

- [Go by Example](https://gobyexample.com/) - Hands-on introduction to Go
- [Go Playground](https://play.golang.org/) - Online Go compiler
- [Go Blog](https://blog.golang.org/) - Official Go blog

### **Video Resources**

- [Gopher Academy](https://www.youtube.com/c/GopherAcademy/) - Go conferences and talks
- [JustForFunc](https://www.youtube.com/c/JustForFunc/) - Go programming videos
- [Go Time](https://changelog.com/gotime/) - Go podcast

---

_This comprehensive guide covers Go from fundamentals to advanced concepts, including real-world examples and FAANG interview questions to help you master Go programming._
