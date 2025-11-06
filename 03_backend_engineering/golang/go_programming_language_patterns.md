---
# Auto-generated front matter
Title: Go Programming Language Patterns
LastUpdated: 2025-11-06T20:45:58.293590
Tags: []
Status: draft
---

# Go Programming Language Patterns - Donovan & Kernighan

## Table of Contents
1. [Introduction](#introduction)
2. [Basic Data Types](#basic-data-types)
3. [Composite Types](#composite-types)
4. [Functions](#functions)
5. [Methods](#methods)
6. [Interfaces](#interfaces)
7. [Goroutines and Channels](#goroutines-and-channels)
8. [Packages and the Go Tool](#packages-and-the-go-tool)
9. [Testing](#testing)
10. [Reflection](#reflection)

## Introduction

This guide is based on "The Go Programming Language" by Alan Donovan and Brian Kernighan. It covers essential Go patterns, idioms, and best practices for writing idiomatic Go code.

### Core Principles
- **Simplicity**: Go favors simplicity over cleverness
- **Composition**: Build complex systems from simple components
- **Concurrency**: Use goroutines and channels for concurrent programming
- **Interfaces**: Use interfaces for abstraction and polymorphism
- **Error Handling**: Explicit error handling, no exceptions

## Basic Data Types

### Numeric Types
```go
// Integer types
var (
    a int8   = 127
    b int16  = 32767
    c int32  = 2147483647
    d int64  = 9223372036854775807
    e int    = 9223372036854775807 // Platform-dependent
)

// Unsigned integer types
var (
    f uint8  = 255
    g uint16 = 65535
    h uint32 = 4294967295
    i uint64 = 18446744073709551615
    j uint   = 18446744073709551615 // Platform-dependent
)

// Floating-point types
var (
    k float32 = 3.14159
    l float64 = 3.141592653589793
)

// Complex types
var (
    m complex64  = 1 + 2i
    n complex128 = 1 + 2i
)

// Type conversions
func convertTypes() {
    var x int = 42
    var y float64 = float64(x)
    var z int = int(y)
    
    // String conversions
    var s string = strconv.Itoa(x)
    var i int, _ = strconv.Atoi(s)
    
    fmt.Printf("x=%d, y=%f, z=%d, s=%s, i=%d\n", x, y, z, s, i)
}
```

### String Operations
```go
// String literals
func stringLiterals() {
    // Raw string literal
    raw := `This is a raw string literal
    that can span multiple lines
    and contains "quotes" without escaping.`
    
    // Interpreted string literal
    interpreted := "This is an interpreted string literal\nwith escape sequences."
    
    fmt.Println(raw)
    fmt.Println(interpreted)
}

// String operations
func stringOperations() {
    s := "Hello, 世界"
    
    // Length
    fmt.Printf("Length: %d\n", len(s))
    
    // Character access
    fmt.Printf("First character: %c\n", s[0])
    
    // Substring
    fmt.Printf("Substring: %s\n", s[0:5])
    
    // String concatenation
    s2 := s + "!"
    fmt.Printf("Concatenated: %s\n", s2)
    
    // String comparison
    if s == "Hello, 世界" {
        fmt.Println("Strings are equal")
    }
}

// String building
func buildString() {
    var builder strings.Builder
    
    for i := 0; i < 10; i++ {
        builder.WriteString(fmt.Sprintf("Item %d\n", i))
    }
    
    result := builder.String()
    fmt.Println(result)
}
```

### Constants
```go
// Untyped constants
const (
    Pi = 3.14159
    E  = 2.71828
)

// Typed constants
const (
    MaxInt8  = 127
    MinInt8  = -128
    MaxUint8 = 255
)

// iota for enumerated constants
type Weekday int

const (
    Sunday Weekday = iota
    Monday
    Tuesday
    Wednesday
    Thursday
    Friday
    Saturday
)

// String method for Weekday
func (d Weekday) String() string {
    days := []string{
        "Sunday", "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday",
    }
    if d < 0 || d >= Weekday(len(days)) {
        return "Invalid day"
    }
    return days[d]
}
```

## Composite Types

### Arrays
```go
// Array declaration and initialization
func arrayExamples() {
    // Zero-initialized array
    var a [3]int
    fmt.Println(a) // [0 0 0]
    
    // Array literal
    var b [3]int = [3]int{1, 2, 3}
    fmt.Println(b) // [1 2 3]
    
    // Array literal with ellipsis
    c := [...]int{1, 2, 3, 4, 5}
    fmt.Println(c) // [1 2 3 4 5]
    
    // Array literal with indices
    d := [...]int{0: 1, 2: 3, 4: 5}
    fmt.Println(d) // [1 0 3 0 5]
    
    // Array comparison
    e := [3]int{1, 2, 3}
    f := [3]int{1, 2, 3}
    fmt.Println(e == f) // true
}

// Array operations
func arrayOperations() {
    a := [...]int{1, 2, 3, 4, 5}
    
    // Length
    fmt.Printf("Length: %d\n", len(a))
    
    // Iteration
    for i, v := range a {
        fmt.Printf("a[%d] = %d\n", i, v)
    }
    
    // Iteration with index only
    for i := range a {
        fmt.Printf("a[%d] = %d\n", i, a[i])
    }
    
    // Iteration with value only
    for _, v := range a {
        fmt.Printf("Value: %d\n", v)
    }
}
```

### Slices
```go
// Slice declaration and initialization
func sliceExamples() {
    // Slice literal
    s := []int{1, 2, 3, 4, 5}
    fmt.Println(s) // [1 2 3 4 5]
    
    // Slice from array
    a := [...]int{1, 2, 3, 4, 5}
    s1 := a[1:4] // [2 3 4]
    s2 := a[:3]  // [1 2 3]
    s3 := a[2:]  // [3 4 5]
    s4 := a[:]   // [1 2 3 4 5]
    
    fmt.Println(s1, s2, s3, s4)
    
    // Make slice
    s5 := make([]int, 5)        // [0 0 0 0 0]
    s6 := make([]int, 5, 10)    // [0 0 0 0 0] with capacity 10
    
    fmt.Println(s5, s6)
}

// Slice operations
func sliceOperations() {
    s := []int{1, 2, 3, 4, 5}
    
    // Length and capacity
    fmt.Printf("Length: %d, Capacity: %d\n", len(s), cap(s))
    
    // Append
    s = append(s, 6, 7, 8)
    fmt.Println(s) // [1 2 3 4 5 6 7 8]
    
    // Append slice
    s2 := []int{9, 10}
    s = append(s, s2...)
    fmt.Println(s) // [1 2 3 4 5 6 7 8 9 10]
    
    // Copy
    s3 := make([]int, len(s))
    copy(s3, s)
    fmt.Println(s3) // [1 2 3 4 5 6 7 8 9 10]
    
    // Delete element (middle)
    i := 3
    s = append(s[:i], s[i+1:]...)
    fmt.Println(s) // [1 2 3 5 6 7 8 9 10]
}

// Slice as stack
type Stack struct {
    items []int
}

func (s *Stack) Push(item int) {
    s.items = append(s.items, item)
}

func (s *Stack) Pop() (int, bool) {
    if len(s.items) == 0 {
        return 0, false
    }
    
    index := len(s.items) - 1
    item := s.items[index]
    s.items = s.items[:index]
    return item, true
}

func (s *Stack) Peek() (int, bool) {
    if len(s.items) == 0 {
        return 0, false
    }
    
    return s.items[len(s.items)-1], true
}
```

### Maps
```go
// Map declaration and initialization
func mapExamples() {
    // Map literal
    m1 := map[string]int{
        "apple":  5,
        "banana": 3,
        "orange": 8,
    }
    fmt.Println(m1)
    
    // Make map
    m2 := make(map[string]int)
    m2["apple"] = 5
    m2["banana"] = 3
    fmt.Println(m2)
    
    // Zero value of map
    var m3 map[string]int
    fmt.Println(m3 == nil) // true
}

// Map operations
func mapOperations() {
    m := map[string]int{
        "apple":  5,
        "banana": 3,
        "orange": 8,
    }
    
    // Access
    fmt.Println(m["apple"]) // 5
    
    // Check if key exists
    if value, ok := m["apple"]; ok {
        fmt.Printf("Apple: %d\n", value)
    }
    
    // Delete
    delete(m, "banana")
    fmt.Println(m) // map[apple:5 orange:8]
    
    // Iteration
    for key, value := range m {
        fmt.Printf("%s: %d\n", key, value)
    }
    
    // Iteration with key only
    for key := range m {
        fmt.Printf("Key: %s\n", key)
    }
}

// Map as set
type Set map[string]bool

func (s Set) Add(item string) {
    s[item] = true
}

func (s Set) Remove(item string) {
    delete(s, item)
}

func (s Set) Contains(item string) bool {
    return s[item]
}

func (s Set) Size() int {
    return len(s)
}
```

### Structs
```go
// Struct declaration
type Person struct {
    Name string
    Age  int
    City string
}

// Struct methods
func (p Person) String() string {
    return fmt.Sprintf("%s (%d) from %s", p.Name, p.Age, p.City)
}

func (p *Person) HaveBirthday() {
    p.Age++
}

// Struct initialization
func structExamples() {
    // Zero value
    var p1 Person
    fmt.Println(p1) // { 0 }
    
    // Struct literal
    p2 := Person{
        Name: "Alice",
        Age:  30,
        City: "New York",
    }
    fmt.Println(p2) // Alice (30) from New York
    
    // Struct literal with field names
    p3 := Person{
        Name: "Bob",
        Age:  25,
    }
    fmt.Println(p3) // Bob (25) from
    
    // Struct literal without field names
    p4 := Person{"Charlie", 35, "London"}
    fmt.Println(p4) // Charlie (35) from London
    
    // Pointer to struct
    p5 := &Person{
        Name: "David",
        Age:  40,
        City: "Paris",
    }
    fmt.Println(p5) // &{David 40 Paris}
}

// Embedded structs
type Address struct {
    Street string
    City   string
    State  string
    Zip    string
}

type Employee struct {
    Person
    Address
    ID     string
    Salary float64
}

func embeddedStructExamples() {
    emp := Employee{
        Person: Person{
            Name: "John",
            Age:  30,
        },
        Address: Address{
            Street: "123 Main St",
            City:   "New York",
            State:  "NY",
            Zip:    "10001",
        },
        ID:     "EMP001",
        Salary: 75000,
    }
    
    // Access embedded fields
    fmt.Println(emp.Name)    // John
    fmt.Println(emp.Street)  // 123 Main St
    fmt.Println(emp.ID)      // EMP001
}
```

## Functions

### Function Declaration
```go
// Basic function
func add(a, b int) int {
    return a + b
}

// Multiple parameters
func multiply(a, b, c int) int {
    return a * b * c
}

// Multiple return values
func divide(a, b int) (int, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}

// Named return values
func divideWithNames(a, b int) (result int, err error) {
    if b == 0 {
        err = errors.New("division by zero")
        return
    }
    result = a / b
    return
}

// Variadic function
func sum(numbers ...int) int {
    total := 0
    for _, num := range numbers {
        total += num
    }
    return total
}

// Function as value
func functionAsValue() {
    // Function variable
    var fn func(int, int) int = add
    result := fn(3, 4)
    fmt.Println(result) // 7
    
    // Anonymous function
    fn2 := func(a, b int) int {
        return a * b
    }
    result2 := fn2(3, 4)
    fmt.Println(result2) // 12
    
    // Function literal
    result3 := func(a, b int) int {
        return a - b
    }(10, 3)
    fmt.Println(result3) // 7
}
```

### Closures
```go
// Closure example
func makeCounter() func() int {
    count := 0
    return func() int {
        count++
        return count
    }
}

func closureExamples() {
    counter1 := makeCounter()
    counter2 := makeCounter()
    
    fmt.Println(counter1()) // 1
    fmt.Println(counter1()) // 2
    fmt.Println(counter2()) // 1
    fmt.Println(counter1()) // 3
}

// Closure with parameters
func makeMultiplier(factor int) func(int) int {
    return func(x int) int {
        return x * factor
    }
}

func closureWithParams() {
    double := makeMultiplier(2)
    triple := makeMultiplier(3)
    
    fmt.Println(double(5)) // 10
    fmt.Println(triple(5)) // 15
}
```

### Defer
```go
// Basic defer
func deferExample() {
    fmt.Println("Start")
    defer fmt.Println("Deferred 1")
    defer fmt.Println("Deferred 2")
    fmt.Println("End")
    // Output:
    // Start
    // End
    // Deferred 2
    // Deferred 1
}

// Defer with file operations
func fileExample() error {
    file, err := os.Open("example.txt")
    if err != nil {
        return err
    }
    defer file.Close() // Will be called when function returns
    
    // Process file
    data := make([]byte, 100)
    _, err = file.Read(data)
    if err != nil {
        return err
    }
    
    fmt.Println(string(data))
    return nil
}

// Defer with panic recovery
func panicExample() {
    defer func() {
        if r := recover(); r != nil {
            fmt.Printf("Recovered from panic: %v\n", r)
        }
    }()
    
    panic("Something went wrong!")
}
```

## Methods

### Method Declaration
```go
// Method on value receiver
type Point struct {
    X, Y float64
}

func (p Point) Distance() float64 {
    return math.Sqrt(p.X*p.X + p.Y*p.Y)
}

// Method on pointer receiver
func (p *Point) Scale(factor float64) {
    p.X *= factor
    p.Y *= factor
}

// Method on non-struct type
type MyInt int

func (m MyInt) IsEven() bool {
    return m%2 == 0
}

func methodExamples() {
    p := Point{3, 4}
    fmt.Println(p.Distance()) // 5
    
    p.Scale(2)
    fmt.Println(p) // {6 8}
    
    num := MyInt(4)
    fmt.Println(num.IsEven()) // true
}
```

### Method Sets
```go
// Method sets and interfaces
type Reader interface {
    Read([]byte) (int, error)
}

type Writer interface {
    Write([]byte) (int, error)
}

type ReadWriter interface {
    Reader
    Writer
}

// Method sets for value and pointer receivers
type Counter struct {
    value int
}

func (c Counter) Value() int {
    return c.value
}

func (c *Counter) Increment() {
    c.value++
}

func methodSetExamples() {
    c := Counter{value: 0}
    
    // Both value and pointer can call value receiver methods
    fmt.Println(c.Value())        // 0
    fmt.Println((&c).Value())     // 0
    
    // Only pointer can call pointer receiver methods
    c.Increment()                  // OK
    (&c).Increment()              // OK
    
    fmt.Println(c.Value())        // 2
}
```

## Interfaces

### Interface Declaration
```go
// Basic interface
type Writer interface {
    Write([]byte) (int, error)
}

// Interface with multiple methods
type ReadWriter interface {
    Read([]byte) (int, error)
    Write([]byte) (int, error)
}

// Interface composition
type ReadWriteCloser interface {
    Reader
    Writer
    Closer
}

type Reader interface {
    Read([]byte) (int, error)
}

type Closer interface {
    Close() error
}

// Empty interface
func processValue(v interface{}) {
    switch x := v.(type) {
    case int:
        fmt.Printf("Integer: %d\n", x)
    case string:
        fmt.Printf("String: %s\n", x)
    case bool:
        fmt.Printf("Boolean: %t\n", x)
    default:
        fmt.Printf("Unknown type: %T\n", x)
    }
}
```

### Interface Implementation
```go
// File implements Writer interface
type File struct {
    name string
    data []byte
}

func (f *File) Write(p []byte) (int, error) {
    f.data = append(f.data, p...)
    return len(p), nil
}

func (f *File) Read(p []byte) (int, error) {
    n := copy(p, f.data)
    return n, nil
}

func (f *File) Close() error {
    return nil
}

// Interface usage
func interfaceExamples() {
    var w Writer = &File{name: "test.txt"}
    
    n, err := w.Write([]byte("Hello, World!"))
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Wrote %d bytes\n", n)
    }
    
    // Type assertion
    if file, ok := w.(*File); ok {
        fmt.Printf("File name: %s\n", file.name)
    }
    
    // Type switch
    switch v := w.(type) {
    case *File:
        fmt.Printf("It's a file: %s\n", v.name)
    case Writer:
        fmt.Println("It's a writer")
    default:
        fmt.Println("Unknown type")
    }
}
```

### Interface Values
```go
// Interface values and nil
func interfaceValues() {
    var w Writer
    fmt.Println(w == nil) // true
    
    var f *File
    w = f
    fmt.Println(w == nil) // false (interface value is not nil)
    
    // Check if interface value is nil
    if w != nil {
        fmt.Println("Interface value is not nil")
    }
}

// Interface satisfaction
type Stringer interface {
    String() string
}

type Person struct {
    Name string
    Age  int
}

func (p Person) String() string {
    return fmt.Sprintf("%s (%d)", p.Name, p.Age)
}

func interfaceSatisfaction() {
    var s Stringer = Person{Name: "Alice", Age: 30}
    fmt.Println(s.String()) // Alice (30)
}
```

## Goroutines and Channels

### Goroutines
```go
// Basic goroutine
func goroutineExample() {
    go func() {
        fmt.Println("Hello from goroutine!")
    }()
    
    time.Sleep(100 * time.Millisecond)
    fmt.Println("Hello from main!")
}

// Goroutine with parameters
func goroutineWithParams() {
    for i := 0; i < 5; i++ {
        go func(id int) {
            fmt.Printf("Goroutine %d\n", id)
        }(i)
    }
    
    time.Sleep(100 * time.Millisecond)
}

// WaitGroup for synchronization
func waitGroupExample() {
    var wg sync.WaitGroup
    
    for i := 0; i < 5; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            fmt.Printf("Goroutine %d\n", id)
        }(i)
    }
    
    wg.Wait()
    fmt.Println("All goroutines completed")
}
```

### Channels
```go
// Basic channel operations
func channelExample() {
    ch := make(chan int)
    
    // Send in goroutine
    go func() {
        ch <- 42
    }()
    
    // Receive in main
    value := <-ch
    fmt.Println(value) // 42
}

// Buffered channels
func bufferedChannelExample() {
    ch := make(chan int, 3)
    
    // Send multiple values
    ch <- 1
    ch <- 2
    ch <- 3
    
    // Receive values
    fmt.Println(<-ch) // 1
    fmt.Println(<-ch) // 2
    fmt.Println(<-ch) // 3
}

// Channel direction
func sendOnly(ch chan<- int) {
    ch <- 42
}

func receiveOnly(ch <-chan int) {
    value := <-ch
    fmt.Println(value)
}

// Select statement
func selectExample() {
    ch1 := make(chan int)
    ch2 := make(chan int)
    
    go func() {
        time.Sleep(100 * time.Millisecond)
        ch1 <- 1
    }()
    
    go func() {
        time.Sleep(200 * time.Millisecond)
        ch2 <- 2
    }()
    
    for i := 0; i < 2; i++ {
        select {
        case msg1 := <-ch1:
            fmt.Printf("Received from ch1: %d\n", msg1)
        case msg2 := <-ch2:
            fmt.Printf("Received from ch2: %d\n", msg2)
        case <-time.After(300 * time.Millisecond):
            fmt.Println("Timeout!")
        }
    }
}
```

### Patterns
```go
// Pipeline pattern
func pipelineExample() {
    // Stage 1: Generate numbers
    numbers := make(chan int)
    go func() {
        defer close(numbers)
        for i := 1; i <= 10; i++ {
            numbers <- i
        }
    }()
    
    // Stage 2: Square numbers
    squares := make(chan int)
    go func() {
        defer close(squares)
        for n := range numbers {
            squares <- n * n
        }
    }()
    
    // Stage 3: Print results
    for s := range squares {
        fmt.Println(s)
    }
}

// Fan-out pattern
func fanOutExample() {
    input := make(chan int)
    
    // Start workers
    for i := 0; i < 3; i++ {
        go worker(i, input)
    }
    
    // Send work
    for i := 1; i <= 10; i++ {
        input <- i
    }
    close(input)
    
    time.Sleep(100 * time.Millisecond)
}

func worker(id int, input <-chan int) {
    for n := range input {
        fmt.Printf("Worker %d processing %d\n", id, n)
        time.Sleep(100 * time.Millisecond)
    }
}

// Fan-in pattern
func fanInExample() {
    ch1 := make(chan int)
    ch2 := make(chan int)
    
    go func() {
        defer close(ch1)
        for i := 1; i <= 5; i++ {
            ch1 <- i
        }
    }()
    
    go func() {
        defer close(ch2)
        for i := 6; i <= 10; i++ {
            ch2 <- i
        }
    }()
    
    // Fan-in
    for n := range merge(ch1, ch2) {
        fmt.Println(n)
    }
}

func merge(ch1, ch2 <-chan int) <-chan int {
    out := make(chan int)
    
    go func() {
        defer close(out)
        for {
            select {
            case n, ok := <-ch1:
                if !ok {
                    ch1 = nil
                } else {
                    out <- n
                }
            case n, ok := <-ch2:
                if !ok {
                    ch2 = nil
                } else {
                    out <- n
                }
            }
            
            if ch1 == nil && ch2 == nil {
                break
            }
        }
    }()
    
    return out
}
```

## Packages and the Go Tool

### Package Structure
```go
// Package declaration
package math

// Exported function
func Add(a, b int) int {
    return a + b
}

// Unexported function
func add(a, b int) int {
    return a + b
}

// Exported variable
var Pi = 3.14159

// Unexported variable
var pi = 3.14159

// Exported type
type Point struct {
    X, Y float64
}

// Unexported type
type point struct {
    x, y float64
}
```

### Package Initialization
```go
// Package-level variables
var (
    config Config
    logger *log.Logger
)

// Init function
func init() {
    config = loadConfig()
    logger = log.New(os.Stdout, "APP: ", log.LstdFlags)
}

func loadConfig() Config {
    // Load configuration
    return Config{}
}

type Config struct {
    // Configuration fields
}
```

### Go Modules
```go
// go.mod file
module github.com/user/project

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/lib/pq v1.10.9
)

require (
    github.com/bytedance/sonic v1.9.1 // indirect
    github.com/chenzhuoyu/base64x v0.0.0-20221115062448-fe3a3abad311 // indirect
    // ... other dependencies
)
```

## Testing

### Basic Testing
```go
// math_test.go
package math

import "testing"

func TestAdd(t *testing.T) {
    result := Add(2, 3)
    expected := 5
    
    if result != expected {
        t.Errorf("Add(2, 3) = %d; expected %d", result, expected)
    }
}

func TestAddTable(t *testing.T) {
    tests := []struct {
        a, b, expected int
    }{
        {2, 3, 5},
        {0, 0, 0},
        {-1, 1, 0},
        {-2, -3, -5},
    }
    
    for _, test := range tests {
        result := Add(test.a, test.b)
        if result != test.expected {
            t.Errorf("Add(%d, %d) = %d; expected %d", test.a, test.b, result, test.expected)
        }
    }
}
```

### Benchmarking
```go
func BenchmarkAdd(b *testing.B) {
    for i := 0; i < b.N; i++ {
        Add(2, 3)
    }
}

func BenchmarkStringConcat(b *testing.B) {
    for i := 0; i < b.N; i++ {
        _ = "Hello" + " " + "World"
    }
}

func BenchmarkStringBuilder(b *testing.B) {
    for i := 0; i < b.N; i++ {
        var builder strings.Builder
        builder.WriteString("Hello")
        builder.WriteString(" ")
        builder.WriteString("World")
        _ = builder.String()
    }
}
```

### Test Coverage
```go
// Run tests with coverage
// go test -cover

// Generate coverage report
// go test -coverprofile=coverage.out
// go tool cover -html=coverage.out

// Test with race detection
// go test -race
```

## Reflection

### Basic Reflection
```go
import "reflect"

func reflectionExample() {
    var x float64 = 3.14159
    
    // Get type
    t := reflect.TypeOf(x)
    fmt.Println(t) // float64
    
    // Get value
    v := reflect.ValueOf(x)
    fmt.Println(v) // 3.14159
    
    // Get kind
    fmt.Println(v.Kind()) // reflect.Float64
    
    // Get interface value
    fmt.Println(v.Interface()) // 3.14159
}

// Type inspection
func inspectType(x interface{}) {
    t := reflect.TypeOf(x)
    v := reflect.ValueOf(x)
    
    fmt.Printf("Type: %s\n", t)
    fmt.Printf("Kind: %s\n", t.Kind())
    fmt.Printf("Value: %v\n", v)
    
    if t.Kind() == reflect.Struct {
        fmt.Printf("Number of fields: %d\n", t.NumField())
        for i := 0; i < t.NumField(); i++ {
            field := t.Field(i)
            value := v.Field(i)
            fmt.Printf("  %s: %v\n", field.Name, value)
        }
    }
}
```

### Value Manipulation
```go
// Setting values
func setValueExample() {
    var x float64 = 3.14159
    
    // Get pointer to x
    v := reflect.ValueOf(&x)
    
    // Get the element that v points to
    elem := v.Elem()
    
    // Set the value
    elem.SetFloat(2.71828)
    
    fmt.Println(x) // 2.71828
}

// Creating new values
func createValueExample() {
    // Create new slice
    sliceType := reflect.SliceOf(reflect.TypeOf(0))
    slice := reflect.MakeSlice(sliceType, 0, 10)
    
    // Append values
    slice = reflect.Append(slice, reflect.ValueOf(1))
    slice = reflect.Append(slice, reflect.ValueOf(2))
    slice = reflect.Append(slice, reflect.ValueOf(3))
    
    fmt.Println(slice.Interface()) // [1 2 3]
}
```

## Conclusion

The Go Programming Language provides a powerful and elegant way to write concurrent, efficient programs. Key patterns and principles:

1. **Simplicity**: Favor simple, clear code over clever solutions
2. **Composition**: Build complex systems from simple components
3. **Concurrency**: Use goroutines and channels for concurrent programming
4. **Interfaces**: Use interfaces for abstraction and polymorphism
5. **Error Handling**: Handle errors explicitly and gracefully
6. **Testing**: Write comprehensive tests and benchmarks
7. **Reflection**: Use reflection judiciously for dynamic programming
8. **Packages**: Organize code into logical packages
9. **Go Modules**: Use modules for dependency management
10. **Idioms**: Follow Go idioms and conventions

By mastering these patterns and principles, you can write idiomatic, efficient, and maintainable Go code.
