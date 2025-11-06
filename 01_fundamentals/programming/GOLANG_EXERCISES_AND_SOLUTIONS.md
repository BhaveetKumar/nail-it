---
# Auto-generated front matter
Title: Golang Exercises And Solutions
LastUpdated: 2025-11-06T20:45:58.763894
Tags: []
Status: draft
---

# 🏋️ Go Exercises and Solutions

> **Practice exercises with detailed solutions to master Go programming**

## 📚 Exercise Categories

1. [Basic Syntax](#basic-syntax-exercises)
2. [Data Structures](#data-structures-exercises)
3. [Functions](#functions-exercises)
4. [Structs and Methods](#structs-and-methods-exercises)
5. [Interfaces](#interfaces-exercises)
6. [Concurrency](#concurrency-exercises)
7. [Error Handling](#error-handling-exercises)
8. [File Operations](#file-operations-exercises)
9. [HTTP and Web](#http-and-web-exercises)
10. [Advanced Topics](#advanced-topics-exercises)

---

## 🔤 Basic Syntax Exercises

### Exercise 1: Variable Operations
**Problem**: Write a program that declares variables of different types and performs basic operations.

**Solution**:
```go
package main

import "fmt"

func main() {
    // Declare variables
    var name string = "John"
    var age int = 25
    var height float64 = 5.9
    var isStudent bool = true
    
    // Short declaration
    city := "New York"
    salary := 50000.0
    
    // Multiple variables
    var a, b, c int = 1, 2, 3
    x, y := 10, 20
    
    // Perform operations
    sum := a + b + c
    product := x * y
    average := (float64(age) + height) / 2
    
    // Print results
    fmt.Printf("Name: %s, Age: %d, Height: %.1f\n", name, age, height)
    fmt.Printf("Is Student: %t, City: %s, Salary: $%.2f\n", isStudent, city, salary)
    fmt.Printf("Sum: %d, Product: %d, Average: %.2f\n", sum, product, average)
}
```

### Exercise 2: String Manipulation
**Problem**: Write functions to reverse a string, count vowels, and check if a string is a palindrome.

**Solution**:
```go
package main

import (
    "fmt"
    "strings"
)

func reverseString(s string) string {
    runes := []rune(s)
    for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
        runes[i], runes[j] = runes[j], runes[i]
    }
    return string(runes)
}

func countVowels(s string) int {
    vowels := "aeiouAEIOU"
    count := 0
    for _, char := range s {
        if strings.ContainsRune(vowels, char) {
            count++
        }
    }
    return count
}

func isPalindrome(s string) bool {
    s = strings.ToLower(strings.ReplaceAll(s, " ", ""))
    return s == reverseString(s)
}

func main() {
    text := "Hello World"
    fmt.Printf("Original: %s\n", text)
    fmt.Printf("Reversed: %s\n", reverseString(text))
    fmt.Printf("Vowels: %d\n", countVowels(text))
    fmt.Printf("Is Palindrome: %t\n", isPalindrome(text))
    
    palindrome := "racecar"
    fmt.Printf("\n'%s' is palindrome: %t\n", palindrome, isPalindrome(palindrome))
}
```

---

## 📊 Data Structures Exercises

### Exercise 3: Array Operations
**Problem**: Write functions to find the maximum, minimum, and average of an array, and reverse it.

**Solution**:
```go
package main

import "fmt"

func findMax(numbers []int) int {
    if len(numbers) == 0 {
        return 0
    }
    max := numbers[0]
    for _, num := range numbers {
        if num > max {
            max = num
        }
    }
    return max
}

func findMin(numbers []int) int {
    if len(numbers) == 0 {
        return 0
    }
    min := numbers[0]
    for _, num := range numbers {
        if num < min {
            min = num
        }
    }
    return min
}

func findAverage(numbers []int) float64 {
    if len(numbers) == 0 {
        return 0
    }
    sum := 0
    for _, num := range numbers {
        sum += num
    }
    return float64(sum) / float64(len(numbers))
}

func reverseArray(numbers []int) []int {
    reversed := make([]int, len(numbers))
    for i, num := range numbers {
        reversed[len(numbers)-1-i] = num
    }
    return reversed
}

func main() {
    numbers := []int{5, 2, 8, 1, 9, 3}
    fmt.Printf("Array: %v\n", numbers)
    fmt.Printf("Max: %d\n", findMax(numbers))
    fmt.Printf("Min: %d\n", findMin(numbers))
    fmt.Printf("Average: %.2f\n", findAverage(numbers))
    fmt.Printf("Reversed: %v\n", reverseArray(numbers))
}
```

### Exercise 4: Map Operations
**Problem**: Create a word frequency counter and a function to find the most common word.

**Solution**:
```go
package main

import (
    "fmt"
    "strings"
)

func wordFrequency(text string) map[string]int {
    words := strings.Fields(strings.ToLower(text))
    frequency := make(map[string]int)
    
    for _, word := range words {
        // Remove punctuation
        word = strings.Trim(word, ".,!?;:")
        frequency[word]++
    }
    
    return frequency
}

func mostCommonWord(frequency map[string]int) (string, int) {
    var mostCommon string
    var maxCount int
    
    for word, count := range frequency {
        if count > maxCount {
            maxCount = count
            mostCommon = word
        }
    }
    
    return mostCommon, maxCount
}

func main() {
    text := "The quick brown fox jumps over the lazy dog. The fox is quick and brown."
    
    frequency := wordFrequency(text)
    fmt.Println("Word Frequency:")
    for word, count := range frequency {
        fmt.Printf("%s: %d\n", word, count)
    }
    
    mostCommon, count := mostCommonWord(frequency)
    fmt.Printf("\nMost common word: '%s' (appears %d times)\n", mostCommon, count)
}
```

---

## 🔧 Functions Exercises

### Exercise 5: Mathematical Functions
**Problem**: Write functions to calculate factorial, Fibonacci sequence, and prime numbers.

**Solution**:
```go
package main

import "fmt"

func factorial(n int) int {
    if n <= 1 {
        return 1
    }
    return n * factorial(n-1)
}

func fibonacci(n int) []int {
    if n <= 0 {
        return []int{}
    }
    if n == 1 {
        return []int{0}
    }
    if n == 2 {
        return []int{0, 1}
    }
    
    fib := []int{0, 1}
    for i := 2; i < n; i++ {
        fib = append(fib, fib[i-1]+fib[i-2])
    }
    return fib
}

func isPrime(n int) bool {
    if n < 2 {
        return false
    }
    for i := 2; i*i <= n; i++ {
        if n%i == 0 {
            return false
        }
    }
    return true
}

func primesUpTo(n int) []int {
    var primes []int
    for i := 2; i <= n; i++ {
        if isPrime(i) {
            primes = append(primes, i)
        }
    }
    return primes
}

func main() {
    // Factorial
    fmt.Printf("Factorial of 5: %d\n", factorial(5))
    
    // Fibonacci
    fmt.Printf("First 10 Fibonacci numbers: %v\n", fibonacci(10))
    
    // Prime numbers
    fmt.Printf("Primes up to 20: %v\n", primesUpTo(20))
    
    // Check if number is prime
    fmt.Printf("Is 17 prime? %t\n", isPrime(17))
    fmt.Printf("Is 15 prime? %t\n", isPrime(15))
}
```

### Exercise 6: Higher-Order Functions
**Problem**: Implement map, filter, and reduce functions for slices.

**Solution**:
```go
package main

import "fmt"

// Map function
func mapInts(slice []int, fn func(int) int) []int {
    result := make([]int, len(slice))
    for i, v := range slice {
        result[i] = fn(v)
    }
    return result
}

// Filter function
func filterInts(slice []int, fn func(int) bool) []int {
    var result []int
    for _, v := range slice {
        if fn(v) {
            result = append(result, v)
        }
    }
    return result
}

// Reduce function
func reduceInts(slice []int, initial int, fn func(int, int) int) int {
    result := initial
    for _, v := range slice {
        result = fn(result, v)
    }
    return result
}

func main() {
    numbers := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    
    // Map: square each number
    squared := mapInts(numbers, func(x int) int { return x * x })
    fmt.Printf("Squared: %v\n", squared)
    
    // Filter: get even numbers
    evens := filterInts(numbers, func(x int) bool { return x%2 == 0 })
    fmt.Printf("Even numbers: %v\n", evens)
    
    // Reduce: sum all numbers
    sum := reduceInts(numbers, 0, func(acc, x int) int { return acc + x })
    fmt.Printf("Sum: %d\n", sum)
    
    // Reduce: product of all numbers
    product := reduceInts(numbers, 1, func(acc, x int) int { return acc * x })
    fmt.Printf("Product: %d\n", product)
}
```

---

## 🏗️ Structs and Methods Exercises

### Exercise 7: Bank Account
**Problem**: Create a BankAccount struct with methods for deposit, withdraw, and balance checking.

**Solution**:
```go
package main

import (
    "fmt"
    "errors"
)

type BankAccount struct {
    accountNumber string
    holderName    string
    balance       float64
}

func NewBankAccount(accountNumber, holderName string) *BankAccount {
    return &BankAccount{
        accountNumber: accountNumber,
        holderName:    holderName,
        balance:       0.0,
    }
}

func (ba *BankAccount) Deposit(amount float64) error {
    if amount <= 0 {
        return errors.New("deposit amount must be positive")
    }
    ba.balance += amount
    return nil
}

func (ba *BankAccount) Withdraw(amount float64) error {
    if amount <= 0 {
        return errors.New("withdrawal amount must be positive")
    }
    if amount > ba.balance {
        return errors.New("insufficient funds")
    }
    ba.balance -= amount
    return nil
}

func (ba *BankAccount) GetBalance() float64 {
    return ba.balance
}

func (ba *BankAccount) GetAccountInfo() string {
    return fmt.Sprintf("Account: %s, Holder: %s, Balance: $%.2f", 
        ba.accountNumber, ba.holderName, ba.balance)
}

func main() {
    // Create account
    account := NewBankAccount("123456789", "John Doe")
    fmt.Println(account.GetAccountInfo())
    
    // Deposit money
    err := account.Deposit(1000.0)
    if err != nil {
        fmt.Printf("Deposit error: %v\n", err)
    } else {
        fmt.Println("Deposited $1000")
        fmt.Println(account.GetAccountInfo())
    }
    
    // Withdraw money
    err = account.Withdraw(300.0)
    if err != nil {
        fmt.Printf("Withdrawal error: %v\n", err)
    } else {
        fmt.Println("Withdrew $300")
        fmt.Println(account.GetAccountInfo())
    }
    
    // Try to withdraw more than balance
    err = account.Withdraw(800.0)
    if err != nil {
        fmt.Printf("Withdrawal error: %v\n", err)
    }
}
```

### Exercise 8: Library Management
**Problem**: Create a Library system with Book and Library structs.

**Solution**:
```go
package main

import (
    "fmt"
    "errors"
)

type Book struct {
    ID       int
    Title    string
    Author   string
    ISBN     string
    IsBorrowed bool
}

type Library struct {
    books []Book
    nextID int
}

func NewLibrary() *Library {
    return &Library{
        books:  make([]Book, 0),
        nextID: 1,
    }
}

func (l *Library) AddBook(title, author, isbn string) {
    book := Book{
        ID:        l.nextID,
        Title:     title,
        Author:    author,
        ISBN:      isbn,
        IsBorrowed: false,
    }
    l.books = append(l.books, book)
    l.nextID++
    fmt.Printf("Added book: %s by %s\n", title, author)
}

func (l *Library) FindBook(title string) *Book {
    for i, book := range l.books {
        if book.Title == title {
            return &l.books[i]
        }
    }
    return nil
}

func (l *Library) BorrowBook(title string) error {
    book := l.FindBook(title)
    if book == nil {
        return errors.New("book not found")
    }
    if book.IsBorrowed {
        return errors.New("book is already borrowed")
    }
    book.IsBorrowed = true
    fmt.Printf("Borrowed: %s\n", title)
    return nil
}

func (l *Library) ReturnBook(title string) error {
    book := l.FindBook(title)
    if book == nil {
        return errors.New("book not found")
    }
    if !book.IsBorrowed {
        return errors.New("book is not borrowed")
    }
    book.IsBorrowed = false
    fmt.Printf("Returned: %s\n", title)
    return nil
}

func (l *Library) ListBooks() {
    fmt.Println("\nLibrary Books:")
    for _, book := range l.books {
        status := "Available"
        if book.IsBorrowed {
            status = "Borrowed"
        }
        fmt.Printf("ID: %d, Title: %s, Author: %s, Status: %s\n", 
            book.ID, book.Title, book.Author, status)
    }
}

func main() {
    library := NewLibrary()
    
    // Add books
    library.AddBook("The Go Programming Language", "Alan Donovan", "978-0134190440")
    library.AddBook("Clean Code", "Robert Martin", "978-0132350884")
    library.AddBook("Design Patterns", "Gang of Four", "978-0201633610")
    
    // List books
    library.ListBooks()
    
    // Borrow a book
    err := library.BorrowBook("The Go Programming Language")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    }
    
    // Try to borrow the same book again
    err = library.BorrowBook("The Go Programming Language")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    }
    
    // Return the book
    err = library.ReturnBook("The Go Programming Language")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    }
    
    // List books again
    library.ListBooks()
}
```

---

## 🔌 Interfaces Exercises

### Exercise 9: Shape Interface
**Problem**: Create a Shape interface and implement it for different geometric shapes.

**Solution**:
```go
package main

import (
    "fmt"
    "math"
)

type Shape interface {
    Area() float64
    Perimeter() float64
}

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

type Circle struct {
    Radius float64
}

func (c Circle) Area() float64 {
    return math.Pi * c.Radius * c.Radius
}

func (c Circle) Perimeter() float64 {
    return 2 * math.Pi * c.Radius
}

type Triangle struct {
    A, B, C float64
}

func (t Triangle) Area() float64 {
    // Using Heron's formula
    s := (t.A + t.B + t.C) / 2
    return math.Sqrt(s * (s - t.A) * (s - t.B) * (s - t.C))
}

func (t Triangle) Perimeter() float64 {
    return t.A + t.B + t.C
}

func printShapeInfo(s Shape) {
    fmt.Printf("Area: %.2f, Perimeter: %.2f\n", s.Area(), s.Perimeter())
}

func main() {
    shapes := []Shape{
        Rectangle{Width: 5, Height: 3},
        Circle{Radius: 4},
        Triangle{A: 3, B: 4, C: 5},
    }
    
    for i, shape := range shapes {
        fmt.Printf("Shape %d: ", i+1)
        printShapeInfo(shape)
    }
}
```

### Exercise 10: Animal Interface
**Problem**: Create an Animal interface with methods for making sounds and moving.

**Solution**:
```go
package main

import "fmt"

type Animal interface {
    MakeSound() string
    Move() string
    GetName() string
}

type Dog struct {
    Name string
}

func (d Dog) MakeSound() string {
    return "Woof! Woof!"
}

func (d Dog) Move() string {
    return "Running on four legs"
}

func (d Dog) GetName() string {
    return d.Name
}

type Bird struct {
    Name string
}

func (b Bird) MakeSound() string {
    return "Tweet! Tweet!"
}

func (b Bird) Move() string {
    return "Flying in the sky"
}

func (b Bird) GetName() string {
    return b.Name
}

type Fish struct {
    Name string
}

func (f Fish) MakeSound() string {
    return "Blub blub"
}

func (f Fish) Move() string {
    return "Swimming in water"
}

func (f Fish) GetName() string {
    return f.Name
}

func describeAnimal(animal Animal) {
    fmt.Printf("%s: %s %s\n", animal.GetName(), animal.MakeSound(), animal.Move())
}

func main() {
    animals := []Animal{
        Dog{Name: "Buddy"},
        Bird{Name: "Tweety"},
        Fish{Name: "Nemo"},
    }
    
    for _, animal := range animals {
        describeAnimal(animal)
    }
}
```

---

## ⚡ Concurrency Exercises

### Exercise 11: Worker Pool
**Problem**: Create a worker pool that processes jobs concurrently.

**Solution**:
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Job struct {
    ID   int
    Data string
}

type Worker struct {
    ID       int
    JobQueue chan Job
    Quit     chan bool
    WG       *sync.WaitGroup
}

func NewWorker(id int, jobQueue chan Job, wg *sync.WaitGroup) *Worker {
    return &Worker{
        ID:       id,
        JobQueue: jobQueue,
        Quit:     make(chan bool),
        WG:       wg,
    }
}

func (w *Worker) Start() {
    go func() {
        for {
            select {
            case job := <-w.JobQueue:
                fmt.Printf("Worker %d processing job %d: %s\n", w.ID, job.ID, job.Data)
                time.Sleep(1 * time.Second) // Simulate work
                fmt.Printf("Worker %d completed job %d\n", w.ID, job.ID)
                w.WG.Done()
            case <-w.Quit:
                fmt.Printf("Worker %d quitting\n", w.ID)
                return
            }
        }
    }()
}

func (w *Worker) Stop() {
    go func() {
        w.Quit <- true
    }()
}

func main() {
    // Create job queue
    jobQueue := make(chan Job, 10)
    
    // Create wait group
    var wg sync.WaitGroup
    
    // Create workers
    numWorkers := 3
    workers := make([]*Worker, numWorkers)
    
    for i := 0; i < numWorkers; i++ {
        workers[i] = NewWorker(i+1, jobQueue, &wg)
        workers[i].Start()
    }
    
    // Create jobs
    jobs := []Job{
        {ID: 1, Data: "Process data 1"},
        {ID: 2, Data: "Process data 2"},
        {ID: 3, Data: "Process data 3"},
        {ID: 4, Data: "Process data 4"},
        {ID: 5, Data: "Process data 5"},
    }
    
    // Send jobs to queue
    for _, job := range jobs {
        wg.Add(1)
        jobQueue <- job
    }
    
    // Wait for all jobs to complete
    wg.Wait()
    fmt.Println("All jobs completed!")
    
    // Stop workers
    for _, worker := range workers {
        worker.Stop()
    }
    
    time.Sleep(1 * time.Second)
    fmt.Println("All workers stopped!")
}
```

### Exercise 12: Producer-Consumer Pattern
**Problem**: Implement a producer-consumer pattern with multiple producers and consumers.

**Solution**:
```go
package main

import (
    "fmt"
    "math/rand"
    "sync"
    "time"
)

type Item struct {
    ID   int
    Data string
}

func producer(id int, items chan<- Item, wg *sync.WaitGroup) {
    defer wg.Done()
    
    for i := 0; i < 5; i++ {
        item := Item{
            ID:   id*100 + i,
            Data: fmt.Sprintf("Producer %d - Item %d", id, i),
        }
        
        items <- item
        fmt.Printf("Producer %d produced item %d\n", id, item.ID)
        
        time.Sleep(time.Duration(rand.Intn(1000)) * time.Millisecond)
    }
}

func consumer(id int, items <-chan Item, wg *sync.WaitGroup) {
    defer wg.Done()
    
    for item := range items {
        fmt.Printf("Consumer %d consumed item %d: %s\n", id, item.ID, item.Data)
        time.Sleep(time.Duration(rand.Intn(1000)) * time.Millisecond)
    }
}

func main() {
    items := make(chan Item, 10)
    var wg sync.WaitGroup
    
    // Start producers
    numProducers := 2
    for i := 0; i < numProducers; i++ {
        wg.Add(1)
        go producer(i+1, items, &wg)
    }
    
    // Start consumers
    numConsumers := 3
    for i := 0; i < numConsumers; i++ {
        wg.Add(1)
        go consumer(i+1, items, &wg)
    }
    
    // Wait for producers to finish
    go func() {
        wg.Wait()
        close(items)
    }()
    
    // Wait for consumers to finish
    wg.Wait()
    fmt.Println("All producers and consumers finished!")
}
```

---

## ❌ Error Handling Exercises

### Exercise 13: Custom Error Types
**Problem**: Create custom error types for different scenarios and handle them appropriately.

**Solution**:
```go
package main

import (
    "fmt"
    "time"
)

type ValidationError struct {
    Field   string
    Message string
    Time    time.Time
}

func (e ValidationError) Error() string {
    return fmt.Sprintf("validation error in field '%s': %s (at %s)", 
        e.Field, e.Message, e.Time.Format("2006-01-02 15:04:05"))
}

type DatabaseError struct {
    Operation string
    Table     string
    Message   string
}

func (e DatabaseError) Error() string {
    return fmt.Sprintf("database error in %s operation on table '%s': %s", 
        e.Operation, e.Table, e.Message)
}

type NetworkError struct {
    URL     string
    Code    int
    Message string
}

func (e NetworkError) Error() string {
    return fmt.Sprintf("network error for URL '%s' (code %d): %s", 
        e.URL, e.Code, e.Message)
}

func validateUser(name, email string) error {
    if name == "" {
        return ValidationError{
            Field:   "name",
            Message: "name cannot be empty",
            Time:    time.Now(),
        }
    }
    
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

func saveToDatabase(table, data string) error {
    if table == "" {
        return DatabaseError{
            Operation: "insert",
            Table:     table,
            Message:   "table name cannot be empty",
        }
    }
    
    if data == "" {
        return DatabaseError{
            Operation: "insert",
            Table:     table,
            Message:   "data cannot be empty",
        }
    }
    
    // Simulate database operation
    return nil
}

func makeHTTPRequest(url string) error {
    if url == "" {
        return NetworkError{
            URL:     url,
            Code:    400,
            Message: "URL cannot be empty",
        }
    }
    
    // Simulate network request
    return nil
}

func handleError(err error) {
    switch e := err.(type) {
    case ValidationError:
        fmt.Printf("Validation Error: %s\n", e.Error())
    case DatabaseError:
        fmt.Printf("Database Error: %s\n", e.Error())
    case NetworkError:
        fmt.Printf("Network Error: %s\n", e.Error())
    default:
        fmt.Printf("Unknown Error: %s\n", err.Error())
    }
}

func main() {
    // Test validation errors
    err := validateUser("", "test@example.com")
    if err != nil {
        handleError(err)
    }
    
    err = validateUser("John", "ab")
    if err != nil {
        handleError(err)
    }
    
    // Test database errors
    err = saveToDatabase("", "some data")
    if err != nil {
        handleError(err)
    }
    
    // Test network errors
    err = makeHTTPRequest("")
    if err != nil {
        handleError(err)
    }
    
    // Test successful operations
    err = validateUser("John", "john@example.com")
    if err != nil {
        handleError(err)
    } else {
        fmt.Println("User validation successful!")
    }
}
```

---

## 📁 File Operations Exercises

### Exercise 14: File Manager
**Problem**: Create a file manager that can create, read, update, and delete files.

**Solution**:
```go
package main

import (
    "bufio"
    "fmt"
    "os"
    "path/filepath"
    "strings"
)

type FileManager struct {
    baseDir string
}

func NewFileManager(baseDir string) *FileManager {
    return &FileManager{baseDir: baseDir}
}

func (fm *FileManager) CreateFile(filename, content string) error {
    filepath := filepath.Join(fm.baseDir, filename)
    
    file, err := os.Create(filepath)
    if err != nil {
        return err
    }
    defer file.Close()
    
    _, err = file.WriteString(content)
    return err
}

func (fm *FileManager) ReadFile(filename string) (string, error) {
    filepath := filepath.Join(fm.baseDir, filename)
    
    content, err := os.ReadFile(filepath)
    if err != nil {
        return "", err
    }
    
    return string(content), nil
}

func (fm *FileManager) UpdateFile(filename, content string) error {
    filepath := filepath.Join(fm.baseDir, filename)
    
    file, err := os.OpenFile(filepath, os.O_WRONLY|os.O_TRUNC, 0644)
    if err != nil {
        return err
    }
    defer file.Close()
    
    _, err = file.WriteString(content)
    return err
}

func (fm *FileManager) DeleteFile(filename string) error {
    filepath := filepath.Join(fm.baseDir, filename)
    return os.Remove(filepath)
}

func (fm *FileManager) ListFiles() ([]string, error) {
    files, err := os.ReadDir(fm.baseDir)
    if err != nil {
        return nil, err
    }
    
    var filenames []string
    for _, file := range files {
        if !file.IsDir() {
            filenames = append(filenames, file.Name())
        }
    }
    
    return filenames, nil
}

func (fm *FileManager) FileExists(filename string) bool {
    filepath := filepath.Join(fm.baseDir, filename)
    _, err := os.Stat(filepath)
    return !os.IsNotExist(err)
}

func main() {
    // Create file manager
    fm := NewFileManager("./test_files")
    
    // Create directory if it doesn't exist
    os.MkdirAll(fm.baseDir, 0755)
    
    // Create a file
    err := fm.CreateFile("test.txt", "Hello, World!\nThis is a test file.")
    if err != nil {
        fmt.Printf("Error creating file: %v\n", err)
        return
    }
    fmt.Println("File created successfully!")
    
    // Read the file
    content, err := fm.ReadFile("test.txt")
    if err != nil {
        fmt.Printf("Error reading file: %v\n", err)
        return
    }
    fmt.Printf("File content:\n%s\n", content)
    
    // Update the file
    err = fm.UpdateFile("test.txt", "Updated content!\nThis file has been modified.")
    if err != nil {
        fmt.Printf("Error updating file: %v\n", err)
        return
    }
    fmt.Println("File updated successfully!")
    
    // Read the updated file
    content, err = fm.ReadFile("test.txt")
    if err != nil {
        fmt.Printf("Error reading file: %v\n", err)
        return
    }
    fmt.Printf("Updated file content:\n%s\n", content)
    
    // List files
    files, err := fm.ListFiles()
    if err != nil {
        fmt.Printf("Error listing files: %v\n", err)
        return
    }
    fmt.Printf("Files in directory: %v\n", files)
    
    // Check if file exists
    exists := fm.FileExists("test.txt")
    fmt.Printf("File exists: %t\n", exists)
    
    // Delete the file
    err = fm.DeleteFile("test.txt")
    if err != nil {
        fmt.Printf("Error deleting file: %v\n", err)
        return
    }
    fmt.Println("File deleted successfully!")
    
    // Check if file exists after deletion
    exists = fm.FileExists("test.txt")
    fmt.Printf("File exists after deletion: %t\n", exists)
}
```

---

## 🌐 HTTP and Web Exercises

### Exercise 15: HTTP Client
**Problem**: Create an HTTP client that can make GET and POST requests and handle responses.

**Solution**:
```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "time"
)

type HTTPClient struct {
    client  *http.Client
    baseURL string
}

func NewHTTPClient(baseURL string) *HTTPClient {
    return &HTTPClient{
        client: &http.Client{
            Timeout: 10 * time.Second,
        },
        baseURL: baseURL,
    }
}

func (c *HTTPClient) Get(path string) ([]byte, error) {
    url := c.baseURL + path
    resp, err := c.client.Get(url)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("HTTP error: %d", resp.StatusCode)
    }
    
    return io.ReadAll(resp.Body)
}

func (c *HTTPClient) Post(path string, data interface{}) ([]byte, error) {
    url := c.baseURL + path
    
    jsonData, err := json.Marshal(data)
    if err != nil {
        return nil, err
    }
    
    resp, err := c.client.Post(url, "application/json", bytes.NewBuffer(jsonData))
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
        return nil, fmt.Errorf("HTTP error: %d", resp.StatusCode)
    }
    
    return io.ReadAll(resp.Body)
}

func (c *HTTPClient) GetJSON(path string, v interface{}) error {
    data, err := c.Get(path)
    if err != nil {
        return err
    }
    
    return json.Unmarshal(data, v)
}

func (c *HTTPClient) PostJSON(path string, data interface{}, result interface{}) error {
    response, err := c.Post(path, data)
    if err != nil {
        return err
    }
    
    if result != nil {
        return json.Unmarshal(response, result)
    }
    
    return nil
}

func main() {
    // Create HTTP client
    client := NewHTTPClient("https://jsonplaceholder.typicode.com")
    
    // GET request
    fmt.Println("Making GET request...")
    data, err := client.Get("/posts/1")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    fmt.Printf("Response: %s\n", string(data))
    
    // GET request with JSON parsing
    fmt.Println("\nMaking GET request with JSON parsing...")
    var post struct {
        ID    int    `json:"id"`
        Title string `json:"title"`
        Body  string `json:"body"`
    }
    
    err = client.GetJSON("/posts/1", &post)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    fmt.Printf("Post: %+v\n", post)
    
    // POST request
    fmt.Println("\nMaking POST request...")
    newPost := map[string]interface{}{
        "title": "My New Post",
        "body":  "This is the content of my new post",
        "userId": 1,
    }
    
    response, err := client.Post("/posts", newPost)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    fmt.Printf("Response: %s\n", string(response))
}
```

---

## 🎯 Advanced Topics Exercises

### Exercise 16: Reflection
**Problem**: Use reflection to inspect and manipulate struct fields dynamically.

**Solution**:
```go
package main

import (
    "fmt"
    "reflect"
)

type Person struct {
    Name    string `json:"name" tag:"required"`
    Age     int    `json:"age" tag:"required"`
    Email   string `json:"email" tag:"optional"`
    Address string `json:"address" tag:"optional"`
}

func inspectStruct(v interface{}) {
    val := reflect.ValueOf(v)
    typ := reflect.TypeOf(v)
    
    fmt.Printf("Type: %s\n", typ.Name())
    fmt.Printf("Kind: %s\n", val.Kind())
    fmt.Printf("NumField: %d\n", val.NumField())
    
    for i := 0; i < val.NumField(); i++ {
        field := val.Field(i)
        fieldType := typ.Field(i)
        
        fmt.Printf("Field %d: %s (%s) = %v\n", 
            i, fieldType.Name, field.Kind(), field.Interface())
        
        // Check tags
        if tag := fieldType.Tag.Get("json"); tag != "" {
            fmt.Printf("  JSON tag: %s\n", tag)
        }
        if tag := fieldType.Tag.Get("tag"); tag != "" {
            fmt.Printf("  Custom tag: %s\n", tag)
        }
    }
}

func setFieldValue(v interface{}, fieldName string, newValue interface{}) error {
    val := reflect.ValueOf(v)
    
    // Check if it's a pointer
    if val.Kind() != reflect.Ptr {
        return fmt.Errorf("value must be a pointer")
    }
    
    // Get the element
    val = val.Elem()
    
    // Find the field
    field := val.FieldByName(fieldName)
    if !field.IsValid() {
        return fmt.Errorf("field %s not found", fieldName)
    }
    
    // Check if field can be set
    if !field.CanSet() {
        return fmt.Errorf("field %s cannot be set", fieldName)
    }
    
    // Set the value
    newVal := reflect.ValueOf(newValue)
    if field.Type() != newVal.Type() {
        return fmt.Errorf("type mismatch: expected %s, got %s", 
            field.Type(), newVal.Type())
    }
    
    field.Set(newVal)
    return nil
}

func getFieldValue(v interface{}, fieldName string) (interface{}, error) {
    val := reflect.ValueOf(v)
    
    // Get the element if it's a pointer
    if val.Kind() == reflect.Ptr {
        val = val.Elem()
    }
    
    // Find the field
    field := val.FieldByName(fieldName)
    if !field.IsValid() {
        return nil, fmt.Errorf("field %s not found", fieldName)
    }
    
    return field.Interface(), nil
}

func main() {
    // Create a person
    person := &Person{
        Name:    "John Doe",
        Age:     30,
        Email:   "john@example.com",
        Address: "123 Main St",
    }
    
    // Inspect the struct
    fmt.Println("=== Inspecting Person Struct ===")
    inspectStruct(person)
    
    // Get field value
    fmt.Println("\n=== Getting Field Values ===")
    name, err := getFieldValue(person, "Name")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Name: %v\n", name)
    }
    
    age, err := getFieldValue(person, "Age")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Age: %v\n", age)
    }
    
    // Set field value
    fmt.Println("\n=== Setting Field Values ===")
    err = setFieldValue(person, "Name", "Jane Doe")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Updated name: %s\n", person.Name)
    }
    
    err = setFieldValue(person, "Age", 25)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Updated age: %d\n", person.Age)
    }
    
    // Inspect the updated struct
    fmt.Println("\n=== Updated Person Struct ===")
    inspectStruct(person)
}
```

---

## 🏆 Exercise Summary

These exercises cover:

1. **Basic Syntax**: Variables, strings, control structures
2. **Data Structures**: Arrays, slices, maps
3. **Functions**: Mathematical functions, higher-order functions
4. **Structs and Methods**: Bank account, library management
5. **Interfaces**: Shape interface, animal interface
6. **Concurrency**: Worker pools, producer-consumer patterns
7. **Error Handling**: Custom error types
8. **File Operations**: File manager
9. **HTTP and Web**: HTTP client
10. **Advanced Topics**: Reflection

Each exercise includes:
- Problem description
- Complete solution
- Detailed explanations
- Best practices
- Error handling

**🚀 Practice these exercises regularly to master Go programming!**
