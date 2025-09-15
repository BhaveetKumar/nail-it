# 🎨 Software Design Patterns for Backend Engineers

## Table of Contents
1. [SOLID Principles](#solid-principles)
2. [Creational Patterns](#creational-patterns)
3. [Structural Patterns](#structural-patterns)
4. [Behavioral Patterns](#behavioral-patterns)
5. [Concurrency Patterns](#concurrency-patterns)
6. [Architectural Patterns](#architectural-patterns)
7. [Go Implementation Examples](#go-implementation-examples)
8. [Interview Questions](#interview-questions)

## SOLID Principles

### Single Responsibility Principle (SRP)
**Definition**: A class should have only one reason to change.

**Example**:
```go
// Bad: Multiple responsibilities
type User struct {
    ID       int
    Name     string
    Email    string
    Password string
}

func (u *User) Save() error {
    // Database operations
}

func (u *User) SendEmail() error {
    // Email operations
}

// Good: Separated responsibilities
type User struct {
    ID    int
    Name  string
    Email string
}

type UserRepository struct{}

func (r *UserRepository) Save(user *User) error {
    // Database operations
}

type EmailService struct{}

func (s *EmailService) SendEmail(user *User) error {
    // Email operations
}
```

### Open/Closed Principle (OCP)
**Definition**: Software entities should be open for extension but closed for modification.

**Example**:
```go
// Bad: Modifying existing code
type PaymentProcessor struct{}

func (p *PaymentProcessor) ProcessPayment(amount float64, method string) error {
    switch method {
    case "credit":
        // Credit card logic
    case "paypal":
        // PayPal logic
    // Adding new payment method requires modification
    }
}

// Good: Open for extension
type PaymentMethod interface {
    Process(amount float64) error
}

type CreditCardPayment struct{}
func (c *CreditCardPayment) Process(amount float64) error { /* ... */ }

type PayPalPayment struct{}
func (p *PayPalPayment) Process(amount float64) error { /* ... */ }

type PaymentProcessor struct{}

func (p *PaymentProcessor) ProcessPayment(amount float64, method PaymentMethod) error {
    return method.Process(amount)
}
```

### Liskov Substitution Principle (LSP)
**Definition**: Objects of a superclass should be replaceable with objects of a subclass.

**Example**:
```go
type Shape interface {
    Area() float64
}

type Rectangle struct {
    Width  float64
    Height float64
}

func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

type Square struct {
    Side float64
}

func (s Square) Area() float64 {
    return s.Side * s.Side
}

// Both Rectangle and Square can be used interchangeably
func CalculateTotalArea(shapes []Shape) float64 {
    total := 0.0
    for _, shape := range shapes {
        total += shape.Area()
    }
    return total
}
```

### Interface Segregation Principle (ISP)
**Definition**: Clients should not be forced to depend on interfaces they don't use.

**Example**:
```go
// Bad: Fat interface
type Worker interface {
    Work()
    Eat()
    Sleep()
}

// Good: Segregated interfaces
type Workable interface {
    Work()
}

type Eatable interface {
    Eat()
}

type Sleepable interface {
    Sleep()
}

type Human struct{}
func (h Human) Work()  { /* ... */ }
func (h Human) Eat()   { /* ... */ }
func (h Human) Sleep() { /* ... */ }

type Robot struct{}
func (r Robot) Work() { /* ... */ }
// Robot doesn't need Eat() or Sleep()
```

### Dependency Inversion Principle (DIP)
**Definition**: High-level modules should not depend on low-level modules. Both should depend on abstractions.

**Example**:
```go
// Bad: High-level depends on low-level
type EmailService struct{}

func (e *EmailService) SendEmail(to, subject, body string) error {
    // Direct email implementation
}

type NotificationService struct {
    emailService *EmailService
}

// Good: Both depend on abstraction
type MessageSender interface {
    Send(to, subject, body string) error
}

type EmailService struct{}
func (e *EmailService) Send(to, subject, body string) error { /* ... */ }

type SMSService struct{}
func (s *SMSService) Send(to, subject, body string) error { /* ... */ }

type NotificationService struct {
    sender MessageSender
}

func (n *NotificationService) Notify(to, subject, body string) error {
    return n.sender.Send(to, subject, body)
}
```

## Creational Patterns

### Singleton Pattern
**Purpose**: Ensure a class has only one instance and provide global access to it.

```go
package main

import (
    "sync"
    "fmt"
)

type Database struct {
    connection string
}

var (
    instance *Database
    once     sync.Once
)

func GetDatabase() *Database {
    once.Do(func() {
        instance = &Database{
            connection: "database_connection_string",
        }
    })
    return instance
}

func main() {
    db1 := GetDatabase()
    db2 := GetDatabase()
    
    fmt.Println(db1 == db2) // true
}
```

### Factory Pattern
**Purpose**: Create objects without specifying their exact class.

```go
package main

import "fmt"

type PaymentMethod interface {
    Process(amount float64) error
}

type CreditCard struct{}
func (c *CreditCard) Process(amount float64) error {
    fmt.Printf("Processing $%.2f with Credit Card\n", amount)
    return nil
}

type PayPal struct{}
func (p *PayPal) Process(amount float64) error {
    fmt.Printf("Processing $%.2f with PayPal\n", amount)
    return nil
}

type PaymentMethodType int

const (
    CreditCardType PaymentMethodType = iota
    PayPalType
)

type PaymentFactory struct{}

func (f *PaymentFactory) CreatePaymentMethod(methodType PaymentMethodType) PaymentMethod {
    switch methodType {
    case CreditCardType:
        return &CreditCard{}
    case PayPalType:
        return &PayPal{}
    default:
        return nil
    }
}

func main() {
    factory := &PaymentFactory{}
    
    creditCard := factory.CreatePaymentMethod(CreditCardType)
    creditCard.Process(100.0)
    
    paypal := factory.CreatePaymentMethod(PayPalType)
    paypal.Process(50.0)
}
```

### Builder Pattern
**Purpose**: Construct complex objects step by step.

```go
package main

import "fmt"

type Computer struct {
    CPU     string
    RAM     int
    Storage string
    GPU     string
}

type ComputerBuilder struct {
    computer *Computer
}

func NewComputerBuilder() *ComputerBuilder {
    return &ComputerBuilder{
        computer: &Computer{},
    }
}

func (b *ComputerBuilder) SetCPU(cpu string) *ComputerBuilder {
    b.computer.CPU = cpu
    return b
}

func (b *ComputerBuilder) SetRAM(ram int) *ComputerBuilder {
    b.computer.RAM = ram
    return b
}

func (b *ComputerBuilder) SetStorage(storage string) *ComputerBuilder {
    b.computer.Storage = storage
    return b
}

func (b *ComputerBuilder) SetGPU(gpu string) *ComputerBuilder {
    b.computer.GPU = gpu
    return b
}

func (b *ComputerBuilder) Build() *Computer {
    return b.computer
}

func main() {
    computer := NewComputerBuilder().
        SetCPU("Intel i7").
        SetRAM(16).
        SetStorage("1TB SSD").
        SetGPU("RTX 3080").
        Build()
    
    fmt.Printf("Computer: %+v\n", computer)
}
```

## Structural Patterns

### Adapter Pattern
**Purpose**: Allow incompatible interfaces to work together.

```go
package main

import "fmt"

// Legacy system
type LegacyPrinter interface {
    PrintLegacy(text string) string
}

type LegacyPrinterImpl struct{}

func (l *LegacyPrinterImpl) PrintLegacy(text string) string {
    return fmt.Sprintf("Legacy: %s", text)
}

// New system
type ModernPrinter interface {
    Print(text string) string
}

type ModernPrinterImpl struct{}

func (m *ModernPrinterImpl) Print(text string) string {
    return fmt.Sprintf("Modern: %s", text)
}

// Adapter
type PrinterAdapter struct {
    legacyPrinter LegacyPrinter
}

func (a *PrinterAdapter) Print(text string) string {
    return a.legacyPrinter.PrintLegacy(text)
}

func main() {
    legacyPrinter := &LegacyPrinterImpl{}
    adapter := &PrinterAdapter{legacyPrinter: legacyPrinter}
    
    result := adapter.Print("Hello World")
    fmt.Println(result) // Output: Legacy: Hello World
}
```

### Decorator Pattern
**Purpose**: Add behavior to objects dynamically.

```go
package main

import "fmt"

type Coffee interface {
    Cost() float64
    Description() string
}

type SimpleCoffee struct{}

func (c *SimpleCoffee) Cost() float64 {
    return 2.0
}

func (c *SimpleCoffee) Description() string {
    return "Simple coffee"
}

type CoffeeDecorator struct {
    coffee Coffee
}

type MilkDecorator struct {
    CoffeeDecorator
}

func (m *MilkDecorator) Cost() float64 {
    return m.coffee.Cost() + 0.5
}

func (m *MilkDecorator) Description() string {
    return m.coffee.Description() + ", milk"
}

type SugarDecorator struct {
    CoffeeDecorator
}

func (s *SugarDecorator) Cost() float64 {
    return s.coffee.Cost() + 0.2
}

func (s *SugarDecorator) Description() string {
    return s.coffee.Description() + ", sugar"
}

func main() {
    coffee := &SimpleCoffee{}
    
    coffeeWithMilk := &MilkDecorator{
        CoffeeDecorator: CoffeeDecorator{coffee: coffee},
    }
    
    coffeeWithMilkAndSugar := &SugarDecorator{
        CoffeeDecorator: CoffeeDecorator{coffee: coffeeWithMilk},
    }
    
    fmt.Printf("Description: %s\n", coffeeWithMilkAndSugar.Description())
    fmt.Printf("Cost: $%.2f\n", coffeeWithMilkAndSugar.Cost())
}
```

### Facade Pattern
**Purpose**: Provide a simplified interface to a complex subsystem.

```go
package main

import "fmt"

// Complex subsystems
type CPU struct{}

func (c *CPU) Start() {
    fmt.Println("CPU started")
}

func (c *CPU) Stop() {
    fmt.Println("CPU stopped")
}

type Memory struct{}

func (m *Memory) Load() {
    fmt.Println("Memory loaded")
}

func (m *Memory) Unload() {
    fmt.Println("Memory unloaded")
}

type HardDrive struct{}

func (h *HardDrive) Read() {
    fmt.Println("Hard drive reading")
}

func (h *HardDrive) Write() {
    fmt.Println("Hard drive writing")
}

// Facade
type Computer struct {
    cpu       *CPU
    memory    *Memory
    hardDrive *HardDrive
}

func NewComputer() *Computer {
    return &Computer{
        cpu:       &CPU{},
        memory:    &Memory{},
        hardDrive: &HardDrive{},
    }
}

func (c *Computer) Start() {
    fmt.Println("Starting computer...")
    c.cpu.Start()
    c.memory.Load()
    c.hardDrive.Read()
    fmt.Println("Computer started successfully!")
}

func (c *Computer) Stop() {
    fmt.Println("Stopping computer...")
    c.cpu.Stop()
    c.memory.Unload()
    c.hardDrive.Write()
    fmt.Println("Computer stopped successfully!")
}

func main() {
    computer := NewComputer()
    computer.Start()
    computer.Stop()
}
```

## Behavioral Patterns

### Observer Pattern
**Purpose**: Define a one-to-many dependency between objects.

```go
package main

import "fmt"

type Observer interface {
    Update(data interface{})
}

type Subject interface {
    Attach(observer Observer)
    Detach(observer Observer)
    Notify()
}

type NewsAgency struct {
    observers []Observer
    news      string
}

func (n *NewsAgency) Attach(observer Observer) {
    n.observers = append(n.observers, observer)
}

func (n *NewsAgency) Detach(observer Observer) {
    for i, obs := range n.observers {
        if obs == observer {
            n.observers = append(n.observers[:i], n.observers[i+1:]...)
            break
        }
    }
}

func (n *NewsAgency) Notify() {
    for _, observer := range n.observers {
        observer.Update(n.news)
    }
}

func (n *NewsAgency) SetNews(news string) {
    n.news = news
    n.Notify()
}

type NewsChannel struct {
    name string
}

func (nc *NewsChannel) Update(data interface{}) {
    fmt.Printf("%s received news: %s\n", nc.name, data)
}

func main() {
    agency := &NewsAgency{}
    
    channel1 := &NewsChannel{name: "CNN"}
    channel2 := &NewsChannel{name: "BBC"}
    
    agency.Attach(channel1)
    agency.Attach(channel2)
    
    agency.SetNews("Breaking: New technology breakthrough!")
}
```

### Strategy Pattern
**Purpose**: Define a family of algorithms and make them interchangeable.

```go
package main

import "fmt"

type SortingStrategy interface {
    Sort(data []int) []int
}

type BubbleSort struct{}

func (b *BubbleSort) Sort(data []int) []int {
    n := len(data)
    for i := 0; i < n-1; i++ {
        for j := 0; j < n-i-1; j++ {
            if data[j] > data[j+1] {
                data[j], data[j+1] = data[j+1], data[j]
            }
        }
    }
    return data
}

type QuickSort struct{}

func (q *QuickSort) Sort(data []int) []int {
    if len(data) < 2 {
        return data
    }
    
    left, right := 0, len(data)-1
    pivot := data[len(data)/2]
    
    for left <= right {
        for data[left] < pivot {
            left++
        }
        for data[right] > pivot {
            right--
        }
        if left <= right {
            data[left], data[right] = data[right], data[left]
            left++
            right--
        }
    }
    
    if right > 0 {
        q.Sort(data[:right+1])
    }
    if left < len(data) {
        q.Sort(data[left:])
    }
    
    return data
}

type Sorter struct {
    strategy SortingStrategy
}

func (s *Sorter) SetStrategy(strategy SortingStrategy) {
    s.strategy = strategy
}

func (s *Sorter) Sort(data []int) []int {
    return s.strategy.Sort(data)
}

func main() {
    data := []int{64, 34, 25, 12, 22, 11, 90}
    
    sorter := &Sorter{}
    
    // Use bubble sort
    sorter.SetStrategy(&BubbleSort{})
    result1 := sorter.Sort(data)
    fmt.Println("Bubble sort:", result1)
    
    // Use quick sort
    data2 := []int{64, 34, 25, 12, 22, 11, 90}
    sorter.SetStrategy(&QuickSort{})
    result2 := sorter.Sort(data2)
    fmt.Println("Quick sort:", result2)
}
```

### Command Pattern
**Purpose**: Encapsulate a request as an object.

```go
package main

import "fmt"

type Command interface {
    Execute()
    Undo()
}

type Light struct {
    isOn bool
}

func (l *Light) TurnOn() {
    l.isOn = true
    fmt.Println("Light is on")
}

func (l *Light) TurnOff() {
    l.isOn = false
    fmt.Println("Light is off")
}

type LightOnCommand struct {
    light *Light
}

func (c *LightOnCommand) Execute() {
    c.light.TurnOn()
}

func (c *LightOnCommand) Undo() {
    c.light.TurnOff()
}

type LightOffCommand struct {
    light *Light
}

func (c *LightOffCommand) Execute() {
    c.light.TurnOff()
}

func (c *LightOffCommand) Undo() {
    c.light.TurnOn()
}

type RemoteControl struct {
    command Command
}

func (r *RemoteControl) SetCommand(command Command) {
    r.command = command
}

func (r *RemoteControl) PressButton() {
    r.command.Execute()
}

func (r *RemoteControl) PressUndo() {
    r.command.Undo()
}

func main() {
    light := &Light{}
    
    lightOnCommand := &LightOnCommand{light: light}
    lightOffCommand := &LightOffCommand{light: light}
    
    remote := &RemoteControl{}
    
    remote.SetCommand(lightOnCommand)
    remote.PressButton()
    
    remote.SetCommand(lightOffCommand)
    remote.PressButton()
    remote.PressUndo()
}
```

## Concurrency Patterns

### Worker Pool Pattern
**Purpose**: Manage a pool of workers to process tasks concurrently.

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Task struct {
    ID   int
    Data string
}

type Worker struct {
    ID       int
    TaskChan chan Task
    QuitChan chan bool
    WG       *sync.WaitGroup
}

func NewWorker(id int, taskChan chan Task, wg *sync.WaitGroup) *Worker {
    return &Worker{
        ID:       id,
        TaskChan: taskChan,
        QuitChan: make(chan bool),
        WG:       wg,
    }
}

func (w *Worker) Start() {
    go func() {
        defer w.WG.Done()
        for {
            select {
            case task := <-w.TaskChan:
                fmt.Printf("Worker %d processing task %d: %s\n", w.ID, task.ID, task.Data)
                time.Sleep(100 * time.Millisecond) // Simulate work
            case <-w.QuitChan:
                fmt.Printf("Worker %d quitting\n", w.ID)
                return
            }
        }
    }()
}

func (w *Worker) Stop() {
    w.QuitChan <- true
}

type WorkerPool struct {
    Workers    []*Worker
    TaskChan   chan Task
    QuitChan   chan bool
    WG         sync.WaitGroup
}

func NewWorkerPool(numWorkers int) *WorkerPool {
    taskChan := make(chan Task, 100)
    
    pool := &WorkerPool{
        Workers:  make([]*Worker, numWorkers),
        TaskChan: taskChan,
        QuitChan: make(chan bool),
    }
    
    for i := 0; i < numWorkers; i++ {
        pool.Workers[i] = NewWorker(i, taskChan, &pool.WG)
        pool.WG.Add(1)
        pool.Workers[i].Start()
    }
    
    return pool
}

func (p *WorkerPool) AddTask(task Task) {
    p.TaskChan <- task
}

func (p *WorkerPool) Stop() {
    close(p.TaskChan)
    for _, worker := range p.Workers {
        worker.Stop()
    }
    p.WG.Wait()
}

func main() {
    pool := NewWorkerPool(3)
    
    // Add tasks
    for i := 0; i < 10; i++ {
        task := Task{
            ID:   i,
            Data: fmt.Sprintf("Task data %d", i),
        }
        pool.AddTask(task)
    }
    
    time.Sleep(2 * time.Second)
    pool.Stop()
}
```

### Producer-Consumer Pattern
**Purpose**: Decouple data production from consumption.

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Item struct {
    ID   int
    Data string
}

type Producer struct {
    ItemChan chan Item
    QuitChan chan bool
}

func NewProducer(itemChan chan Item) *Producer {
    return &Producer{
        ItemChan: itemChan,
        QuitChan: make(chan bool),
    }
}

func (p *Producer) Start() {
    go func() {
        id := 0
        for {
            select {
            case <-p.QuitChan:
                return
            default:
                item := Item{
                    ID:   id,
                    Data: fmt.Sprintf("Item %d", id),
                }
                p.ItemChan <- item
                fmt.Printf("Produced: %+v\n", item)
                id++
                time.Sleep(500 * time.Millisecond)
            }
        }
    }()
}

func (p *Producer) Stop() {
    p.QuitChan <- true
}

type Consumer struct {
    ID       int
    ItemChan chan Item
    QuitChan chan bool
}

func NewConsumer(id int, itemChan chan Item) *Consumer {
    return &Consumer{
        ID:       id,
        ItemChan: itemChan,
        QuitChan: make(chan bool),
    }
}

func (c *Consumer) Start() {
    go func() {
        for {
            select {
            case item := <-c.ItemChan:
                fmt.Printf("Consumer %d consumed: %+v\n", c.ID, item)
                time.Sleep(1 * time.Second)
            case <-c.QuitChan:
                return
            }
        }
    }()
}

func (c *Consumer) Stop() {
    c.QuitChan <- true
}

func main() {
    itemChan := make(chan Item, 10)
    
    producer := NewProducer(itemChan)
    producer.Start()
    
    consumers := make([]*Consumer, 3)
    for i := 0; i < 3; i++ {
        consumers[i] = NewConsumer(i, itemChan)
        consumers[i].Start()
    }
    
    time.Sleep(5 * time.Second)
    
    producer.Stop()
    for _, consumer := range consumers {
        consumer.Stop()
    }
}
```

## Architectural Patterns

### MVC (Model-View-Controller)
**Purpose**: Separate concerns in user interface applications.

```go
package main

import "fmt"

// Model
type User struct {
    ID   int
    Name string
}

type UserModel struct {
    users []User
}

func (m *UserModel) AddUser(user User) {
    m.users = append(m.users, user)
}

func (m *UserModel) GetUsers() []User {
    return m.users
}

// View
type UserView struct{}

func (v *UserView) DisplayUsers(users []User) {
    fmt.Println("Users:")
    for _, user := range users {
        fmt.Printf("ID: %d, Name: %s\n", user.ID, user.Name)
    }
}

// Controller
type UserController struct {
    model *UserModel
    view  *UserView
}

func NewUserController(model *UserModel, view *UserView) *UserController {
    return &UserController{
        model: model,
        view:  view,
    }
}

func (c *UserController) AddUser(id int, name string) {
    user := User{ID: id, Name: name}
    c.model.AddUser(user)
}

func (c *UserController) DisplayUsers() {
    users := c.model.GetUsers()
    c.view.DisplayUsers(users)
}

func main() {
    model := &UserModel{}
    view := &UserView{}
    controller := NewUserController(model, view)
    
    controller.AddUser(1, "John Doe")
    controller.AddUser(2, "Jane Smith")
    controller.DisplayUsers()
}
```

## Interview Questions

### Basic Concepts
1. **What are the SOLID principles?**
2. **Explain the difference between composition and inheritance.**
3. **What is the purpose of design patterns?**
4. **When would you use the Singleton pattern?**
5. **Explain the Factory pattern and its benefits.**

### Advanced Topics
1. **How do you implement the Observer pattern in Go?**
2. **What are the advantages and disadvantages of the Singleton pattern?**
3. **How would you implement a thread-safe Singleton?**
4. **Explain the Strategy pattern with a real-world example.**
5. **What is the difference between the Adapter and Decorator patterns?**

### System Design
1. **How would you design a notification system using design patterns?**
2. **Design a caching system using appropriate patterns.**
3. **How would you implement a plugin architecture?**
4. **Design a command processing system.**
5. **How would you implement a state machine using patterns?**

## Conclusion

Design patterns are essential tools for creating maintainable, scalable, and robust software. Understanding these patterns helps in:

- Writing cleaner, more organized code
- Solving common design problems
- Communicating design decisions effectively
- Preparing for technical interviews
- Building enterprise-grade applications

Practice implementing these patterns in your projects and understand when to apply each one. Remember that patterns are guidelines, not rigid rules, and should be adapted to your specific use case.
