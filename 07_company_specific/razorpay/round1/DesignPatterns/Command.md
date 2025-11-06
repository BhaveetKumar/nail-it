---
# Auto-generated front matter
Title: Command
LastUpdated: 2025-11-06T20:45:58.517648
Tags: []
Status: draft
---

# Command Pattern

## Pattern Name & Intent

**Command** is a behavioral design pattern that turns a request into a stand-alone object containing all information about the request. This transformation lets you pass requests as method arguments, delay or queue a request's execution, and support undoable operations.

**Key Intent:**

- Encapsulate a request as an object
- Decouple invoker from receiver
- Support undo/redo operations
- Enable queuing and logging of operations
- Support macro commands (composite commands)
- Allow parameterization of objects with different requests

## When to Use

**Use Command when:**

1. **Undo/Redo Operations**: Need to support undo and redo functionality
2. **Request Queuing**: Want to queue operations for later execution
3. **Request Logging**: Need to log operations for auditing or replay
4. **Macro Operations**: Want to compose complex operations from simpler ones
5. **Decoupling**: Need to decouple invoker from receiver
6. **Remote Procedure Calls**: Implementing RPC or API calls as objects
7. **Transactional Operations**: Need to group operations into transactions

**Don't use when:**

- Simple direct method calls are sufficient
- No need for undo/redo functionality
- Operations are not complex enough to warrant encapsulation
- Performance is critical (adds overhead)

## Real-World Use Cases (Payments/Fintech)

### 1. Transaction Management System

```go
// Commands for different transaction operations
type TransactionCommand interface {
    Execute() error
    Undo() error
    GetDescription() string
    GetTransactionID() string
}

// Transfer money command
type TransferMoneyCommand struct {
    fromAccount   string
    toAccount     string
    amount        decimal.Decimal
    transactionID string
    accountService AccountService
    executed      bool
}

func (t *TransferMoneyCommand) Execute() error {
    err := t.accountService.TransferMoney(t.fromAccount, t.toAccount, t.amount, t.transactionID)
    if err == nil {
        t.executed = true
    }
    return err
}

func (t *TransferMoneyCommand) Undo() error {
    if !t.executed {
        return fmt.Errorf("cannot undo: command not executed")
    }
    // Reverse the transfer
    return t.accountService.TransferMoney(t.toAccount, t.fromAccount, t.amount, t.transactionID+"_REVERSAL")
}

// Credit account command
type CreditAccountCommand struct {
    accountID     string
    amount        decimal.Decimal
    transactionID string
    accountService AccountService
    executed      bool
}

func (c *CreditAccountCommand) Execute() error {
    err := c.accountService.CreditAccount(c.accountID, c.amount, c.transactionID)
    if err == nil {
        c.executed = true
    }
    return err
}

func (c *CreditAccountCommand) Undo() error {
    if !c.executed {
        return fmt.Errorf("cannot undo: command not executed")
    }
    // Debit the same amount
    return c.accountService.DebitAccount(c.accountID, c.amount, c.transactionID+"_REVERSAL")
}
```

### 2. Order Management System

```go
// Commands for order operations
type OrderCommand interface {
    Execute() error
    Undo() error
    GetOrderID() string
    GetTimestamp() time.Time
}

// Place order command
type PlaceOrderCommand struct {
    order         *Order
    orderService  OrderService
    inventoryService InventoryService
    paymentService PaymentService
    orderID       string
    timestamp     time.Time
    executed      bool
}

func (p *PlaceOrderCommand) Execute() error {
    // Reserve inventory
    if err := p.inventoryService.ReserveItems(p.order.Items); err != nil {
        return err
    }

    // Process payment
    if err := p.paymentService.ProcessPayment(p.order.PaymentInfo); err != nil {
        p.inventoryService.ReleaseReservation(p.order.Items)
        return err
    }

    // Create order
    if err := p.orderService.CreateOrder(p.order); err != nil {
        p.paymentService.RefundPayment(p.order.PaymentInfo.TransactionID)
        p.inventoryService.ReleaseReservation(p.order.Items)
        return err
    }

    p.executed = true
    return nil
}

func (p *PlaceOrderCommand) Undo() error {
    if !p.executed {
        return fmt.Errorf("cannot undo: command not executed")
    }

    // Cancel order, refund payment, release inventory
    p.orderService.CancelOrder(p.orderID)
    p.paymentService.RefundPayment(p.order.PaymentInfo.TransactionID)
    p.inventoryService.ReleaseReservation(p.order.Items)

    return nil
}

// Cancel order command
type CancelOrderCommand struct {
    orderID       string
    orderService  OrderService
    paymentService PaymentService
    timestamp     time.Time
    executed      bool
}

func (c *CancelOrderCommand) Execute() error {
    order, err := c.orderService.GetOrder(c.orderID)
    if err != nil {
        return err
    }

    if err := c.orderService.CancelOrder(c.orderID); err != nil {
        return err
    }

    if err := c.paymentService.RefundPayment(order.PaymentInfo.TransactionID); err != nil {
        // Try to restore order if refund fails
        c.orderService.RestoreOrder(c.orderID)
        return err
    }

    c.executed = true
    return nil
}
```

### 3. Trading Platform Commands

```go
// Commands for trading operations
type TradingCommand interface {
    Execute() error
    Undo() error
    GetSymbol() string
    GetOrderType() string
}

// Buy order command
type BuyOrderCommand struct {
    symbol        string
    quantity      int64
    price         decimal.Decimal
    orderType     string
    tradingService TradingService
    orderID       string
    executed      bool
}

func (b *BuyOrderCommand) Execute() error {
    orderID, err := b.tradingService.PlaceBuyOrder(b.symbol, b.quantity, b.price, b.orderType)
    if err != nil {
        return err
    }

    b.orderID = orderID
    b.executed = true
    return nil
}

func (b *BuyOrderCommand) Undo() error {
    if !b.executed {
        return fmt.Errorf("cannot undo: command not executed")
    }

    // Cancel the order
    return b.tradingService.CancelOrder(b.orderID)
}

// Sell order command
type SellOrderCommand struct {
    symbol        string
    quantity      int64
    price         decimal.Decimal
    orderType     string
    tradingService TradingService
    orderID       string
    executed      bool
}

func (s *SellOrderCommand) Execute() error {
    orderID, err := s.tradingService.PlaceSellOrder(s.symbol, s.quantity, s.price, s.orderType)
    if err != nil {
        return err
    }

    s.orderID = orderID
    s.executed = true
    return nil
}

func (s *SellOrderCommand) Undo() error {
    if !s.executed {
        return fmt.Errorf("cannot undo: command not executed")
    }

    return s.tradingService.CancelOrder(s.orderID)
}
```

### 4. Risk Management Commands

```go
// Commands for risk management operations
type RiskCommand interface {
    Execute() error
    Undo() error
    GetRiskLevel() string
    GetDescription() string
}

// Set position limit command
type SetPositionLimitCommand struct {
    userID       string
    symbol       string
    limit        decimal.Decimal
    riskService  RiskService
    previousLimit decimal.Decimal
    executed     bool
}

func (s *SetPositionLimitCommand) Execute() error {
    // Get current limit for undo
    currentLimit, err := s.riskService.GetPositionLimit(s.userID, s.symbol)
    if err != nil {
        return err
    }
    s.previousLimit = currentLimit

    // Set new limit
    err = s.riskService.SetPositionLimit(s.userID, s.symbol, s.limit)
    if err == nil {
        s.executed = true
    }
    return err
}

func (s *SetPositionLimitCommand) Undo() error {
    if !s.executed {
        return fmt.Errorf("cannot undo: command not executed")
    }

    // Restore previous limit
    return s.riskService.SetPositionLimit(s.userID, s.symbol, s.previousLimit)
}
```

## Go Implementation

```go
package main

import (
    "fmt"
    "log"
    "time"
    "sync"
    "github.com/shopspring/decimal"
    "github.com/google/uuid"
)

// Command interface
type Command interface {
    Execute() error
    Undo() error
    GetID() string
    GetDescription() string
    GetTimestamp() time.Time
    IsExecuted() bool
}

// Base command implementation
type BaseCommand struct {
    ID          string    `json:"id"`
    Description string    `json:"description"`
    Timestamp   time.Time `json:"timestamp"`
    Executed    bool      `json:"executed"`
}

func NewBaseCommand(description string) BaseCommand {
    return BaseCommand{
        ID:          uuid.New().String(),
        Description: description,
        Timestamp:   time.Now(),
        Executed:    false,
    }
}

func (b *BaseCommand) GetID() string          { return b.ID }
func (b *BaseCommand) GetDescription() string { return b.Description }
func (b *BaseCommand) GetTimestamp() time.Time { return b.Timestamp }
func (b *BaseCommand) IsExecuted() bool       { return b.Executed }

// Account service interface (receiver)
type AccountService interface {
    GetBalance(accountID string) (decimal.Decimal, error)
    CreditAccount(accountID string, amount decimal.Decimal, transactionID string) error
    DebitAccount(accountID string, amount decimal.Decimal, transactionID string) error
    TransferMoney(fromAccount, toAccount string, amount decimal.Decimal, transactionID string) error
    FreezeAccount(accountID string) error
    UnfreezeAccount(accountID string) error
}

// Concrete account service
type BankAccountService struct {
    accounts map[string]*Account
    mu       sync.RWMutex
}

type Account struct {
    ID       string          `json:"id"`
    Balance  decimal.Decimal `json:"balance"`
    Frozen   bool            `json:"frozen"`
    History  []Transaction   `json:"history"`
}

type Transaction struct {
    ID            string          `json:"id"`
    Type          string          `json:"type"`
    Amount        decimal.Decimal `json:"amount"`
    Description   string          `json:"description"`
    Timestamp     time.Time       `json:"timestamp"`
    RunningBalance decimal.Decimal `json:"running_balance"`
}

func NewBankAccountService() *BankAccountService {
    return &BankAccountService{
        accounts: make(map[string]*Account),
    }
}

func (b *BankAccountService) CreateAccount(accountID string, initialBalance decimal.Decimal) {
    b.mu.Lock()
    defer b.mu.Unlock()

    b.accounts[accountID] = &Account{
        ID:      accountID,
        Balance: initialBalance,
        Frozen:  false,
        History: make([]Transaction, 0),
    }
}

func (b *BankAccountService) GetBalance(accountID string) (decimal.Decimal, error) {
    b.mu.RLock()
    defer b.mu.RUnlock()

    account, exists := b.accounts[accountID]
    if !exists {
        return decimal.Zero, fmt.Errorf("account not found: %s", accountID)
    }

    return account.Balance, nil
}

func (b *BankAccountService) CreditAccount(accountID string, amount decimal.Decimal, transactionID string) error {
    b.mu.Lock()
    defer b.mu.Unlock()

    account, exists := b.accounts[accountID]
    if !exists {
        return fmt.Errorf("account not found: %s", accountID)
    }

    if account.Frozen {
        return fmt.Errorf("account is frozen: %s", accountID)
    }

    account.Balance = account.Balance.Add(amount)

    transaction := Transaction{
        ID:             transactionID,
        Type:           "CREDIT",
        Amount:         amount,
        Description:    fmt.Sprintf("Credit %s", amount.String()),
        Timestamp:      time.Now(),
        RunningBalance: account.Balance,
    }

    account.History = append(account.History, transaction)

    log.Printf("Credited %s to account %s. New balance: %s",
        amount.String(), accountID, account.Balance.String())

    return nil
}

func (b *BankAccountService) DebitAccount(accountID string, amount decimal.Decimal, transactionID string) error {
    b.mu.Lock()
    defer b.mu.Unlock()

    account, exists := b.accounts[accountID]
    if !exists {
        return fmt.Errorf("account not found: %s", accountID)
    }

    if account.Frozen {
        return fmt.Errorf("account is frozen: %s", accountID)
    }

    if account.Balance.LessThan(amount) {
        return fmt.Errorf("insufficient funds in account %s", accountID)
    }

    account.Balance = account.Balance.Sub(amount)

    transaction := Transaction{
        ID:             transactionID,
        Type:           "DEBIT",
        Amount:         amount,
        Description:    fmt.Sprintf("Debit %s", amount.String()),
        Timestamp:      time.Now(),
        RunningBalance: account.Balance,
    }

    account.History = append(account.History, transaction)

    log.Printf("Debited %s from account %s. New balance: %s",
        amount.String(), accountID, account.Balance.String())

    return nil
}

func (b *BankAccountService) TransferMoney(fromAccount, toAccount string, amount decimal.Decimal, transactionID string) error {
    // Debit from source account
    if err := b.DebitAccount(fromAccount, amount, transactionID+"_DEBIT"); err != nil {
        return err
    }

    // Credit to destination account
    if err := b.CreditAccount(toAccount, amount, transactionID+"_CREDIT"); err != nil {
        // Rollback debit if credit fails
        b.CreditAccount(fromAccount, amount, transactionID+"_ROLLBACK")
        return err
    }

    log.Printf("Transferred %s from %s to %s", amount.String(), fromAccount, toAccount)
    return nil
}

func (b *BankAccountService) FreezeAccount(accountID string) error {
    b.mu.Lock()
    defer b.mu.Unlock()

    account, exists := b.accounts[accountID]
    if !exists {
        return fmt.Errorf("account not found: %s", accountID)
    }

    account.Frozen = true
    log.Printf("Account %s frozen", accountID)
    return nil
}

func (b *BankAccountService) UnfreezeAccount(accountID string) error {
    b.mu.Lock()
    defer b.mu.Unlock()

    account, exists := b.accounts[accountID]
    if !exists {
        return fmt.Errorf("account not found: %s", accountID)
    }

    account.Frozen = false
    log.Printf("Account %s unfrozen", accountID)
    return nil
}

// Concrete Commands

// Credit Account Command
type CreditAccountCommand struct {
    BaseCommand
    accountService AccountService
    accountID      string
    amount         decimal.Decimal
    transactionID  string
}

func NewCreditAccountCommand(accountService AccountService, accountID string, amount decimal.Decimal) *CreditAccountCommand {
    return &CreditAccountCommand{
        BaseCommand:    NewBaseCommand(fmt.Sprintf("Credit %s to account %s", amount.String(), accountID)),
        accountService: accountService,
        accountID:      accountID,
        amount:         amount,
        transactionID:  uuid.New().String(),
    }
}

func (c *CreditAccountCommand) Execute() error {
    err := c.accountService.CreditAccount(c.accountID, c.amount, c.transactionID)
    if err == nil {
        c.Executed = true
    }
    return err
}

func (c *CreditAccountCommand) Undo() error {
    if !c.Executed {
        return fmt.Errorf("cannot undo: command not executed")
    }

    return c.accountService.DebitAccount(c.accountID, c.amount, c.transactionID+"_UNDO")
}

// Debit Account Command
type DebitAccountCommand struct {
    BaseCommand
    accountService AccountService
    accountID      string
    amount         decimal.Decimal
    transactionID  string
}

func NewDebitAccountCommand(accountService AccountService, accountID string, amount decimal.Decimal) *DebitAccountCommand {
    return &DebitAccountCommand{
        BaseCommand:    NewBaseCommand(fmt.Sprintf("Debit %s from account %s", amount.String(), accountID)),
        accountService: accountService,
        accountID:      accountID,
        amount:         amount,
        transactionID:  uuid.New().String(),
    }
}

func (d *DebitAccountCommand) Execute() error {
    err := d.accountService.DebitAccount(d.accountID, d.amount, d.transactionID)
    if err == nil {
        d.Executed = true
    }
    return err
}

func (d *DebitAccountCommand) Undo() error {
    if !d.Executed {
        return fmt.Errorf("cannot undo: command not executed")
    }

    return d.accountService.CreditAccount(d.accountID, d.amount, d.transactionID+"_UNDO")
}

// Transfer Money Command
type TransferMoneyCommand struct {
    BaseCommand
    accountService AccountService
    fromAccount    string
    toAccount      string
    amount         decimal.Decimal
    transactionID  string
}

func NewTransferMoneyCommand(accountService AccountService, fromAccount, toAccount string, amount decimal.Decimal) *TransferMoneyCommand {
    return &TransferMoneyCommand{
        BaseCommand:    NewBaseCommand(fmt.Sprintf("Transfer %s from %s to %s", amount.String(), fromAccount, toAccount)),
        accountService: accountService,
        fromAccount:    fromAccount,
        toAccount:      toAccount,
        amount:         amount,
        transactionID:  uuid.New().String(),
    }
}

func (t *TransferMoneyCommand) Execute() error {
    err := t.accountService.TransferMoney(t.fromAccount, t.toAccount, t.amount, t.transactionID)
    if err == nil {
        t.Executed = true
    }
    return err
}

func (t *TransferMoneyCommand) Undo() error {
    if !t.Executed {
        return fmt.Errorf("cannot undo: command not executed")
    }

    // Reverse the transfer
    return t.accountService.TransferMoney(t.toAccount, t.fromAccount, t.amount, t.transactionID+"_UNDO")
}

// Freeze Account Command
type FreezeAccountCommand struct {
    BaseCommand
    accountService AccountService
    accountID      string
}

func NewFreezeAccountCommand(accountService AccountService, accountID string) *FreezeAccountCommand {
    return &FreezeAccountCommand{
        BaseCommand:    NewBaseCommand(fmt.Sprintf("Freeze account %s", accountID)),
        accountService: accountService,
        accountID:      accountID,
    }
}

func (f *FreezeAccountCommand) Execute() error {
    err := f.accountService.FreezeAccount(f.accountID)
    if err == nil {
        f.Executed = true
    }
    return err
}

func (f *FreezeAccountCommand) Undo() error {
    if !f.Executed {
        return fmt.Errorf("cannot undo: command not executed")
    }

    return f.accountService.UnfreezeAccount(f.accountID)
}

// Macro Command (Composite Command)
type MacroCommand struct {
    BaseCommand
    commands []Command
}

func NewMacroCommand(description string, commands ...Command) *MacroCommand {
    return &MacroCommand{
        BaseCommand: NewBaseCommand(description),
        commands:    commands,
    }
}

func (m *MacroCommand) Execute() error {
    for i, cmd := range m.commands {
        if err := cmd.Execute(); err != nil {
            // Rollback previously executed commands
            for j := i - 1; j >= 0; j-- {
                if rollbackErr := m.commands[j].Undo(); rollbackErr != nil {
                    log.Printf("Rollback failed for command %s: %v", m.commands[j].GetID(), rollbackErr)
                }
            }
            return fmt.Errorf("macro command failed at step %d: %w", i+1, err)
        }
    }

    m.Executed = true
    return nil
}

func (m *MacroCommand) Undo() error {
    if !m.Executed {
        return fmt.Errorf("cannot undo: macro command not executed")
    }

    // Undo commands in reverse order
    for i := len(m.commands) - 1; i >= 0; i-- {
        if err := m.commands[i].Undo(); err != nil {
            log.Printf("Failed to undo command %s: %v", m.commands[i].GetID(), err)
            return err
        }
    }

    return nil
}

func (m *MacroCommand) GetCommands() []Command {
    return m.commands
}

// Command Invoker
type CommandInvoker struct {
    history    []Command
    currentPos int
    mu         sync.Mutex
}

func NewCommandInvoker() *CommandInvoker {
    return &CommandInvoker{
        history:    make([]Command, 0),
        currentPos: -1,
    }
}

func (ci *CommandInvoker) ExecuteCommand(cmd Command) error {
    ci.mu.Lock()
    defer ci.mu.Unlock()

    err := cmd.Execute()
    if err != nil {
        return err
    }

    // Remove any commands after current position (for redo functionality)
    ci.history = ci.history[:ci.currentPos+1]

    // Add new command to history
    ci.history = append(ci.history, cmd)
    ci.currentPos++

    log.Printf("Executed command: %s (ID: %s)", cmd.GetDescription(), cmd.GetID())
    return nil
}

func (ci *CommandInvoker) Undo() error {
    ci.mu.Lock()
    defer ci.mu.Unlock()

    if ci.currentPos < 0 {
        return fmt.Errorf("no commands to undo")
    }

    cmd := ci.history[ci.currentPos]
    if !cmd.IsExecuted() {
        return fmt.Errorf("command not executed: %s", cmd.GetID())
    }

    err := cmd.Undo()
    if err != nil {
        return fmt.Errorf("undo failed: %w", err)
    }

    ci.currentPos--
    log.Printf("Undid command: %s (ID: %s)", cmd.GetDescription(), cmd.GetID())
    return nil
}

func (ci *CommandInvoker) Redo() error {
    ci.mu.Lock()
    defer ci.mu.Unlock()

    if ci.currentPos >= len(ci.history)-1 {
        return fmt.Errorf("no commands to redo")
    }

    ci.currentPos++
    cmd := ci.history[ci.currentPos]

    err := cmd.Execute()
    if err != nil {
        ci.currentPos-- // Rollback position if execution fails
        return fmt.Errorf("redo failed: %w", err)
    }

    log.Printf("Redid command: %s (ID: %s)", cmd.GetDescription(), cmd.GetID())
    return nil
}

func (ci *CommandInvoker) GetHistory() []Command {
    ci.mu.Lock()
    defer ci.mu.Unlock()

    // Return copy of history up to current position
    result := make([]Command, ci.currentPos+1)
    copy(result, ci.history[:ci.currentPos+1])
    return result
}

func (ci *CommandInvoker) GetHistorySize() int {
    ci.mu.Lock()
    defer ci.mu.Unlock()
    return ci.currentPos + 1
}

func (ci *CommandInvoker) CanUndo() bool {
    ci.mu.Lock()
    defer ci.mu.Unlock()
    return ci.currentPos >= 0
}

func (ci *CommandInvoker) CanRedo() bool {
    ci.mu.Lock()
    defer ci.mu.Unlock()
    return ci.currentPos < len(ci.history)-1
}

// Command Queue for delayed execution
type CommandQueue struct {
    commands []Command
    mu       sync.Mutex
}

func NewCommandQueue() *CommandQueue {
    return &CommandQueue{
        commands: make([]Command, 0),
    }
}

func (cq *CommandQueue) Enqueue(cmd Command) {
    cq.mu.Lock()
    defer cq.mu.Unlock()
    cq.commands = append(cq.commands, cmd)
}

func (cq *CommandQueue) Dequeue() (Command, error) {
    cq.mu.Lock()
    defer cq.mu.Unlock()

    if len(cq.commands) == 0 {
        return nil, fmt.Errorf("queue is empty")
    }

    cmd := cq.commands[0]
    cq.commands = cq.commands[1:]
    return cmd, nil
}

func (cq *CommandQueue) Size() int {
    cq.mu.Lock()
    defer cq.mu.Unlock()
    return len(cq.commands)
}

func (cq *CommandQueue) ExecuteAll(invoker *CommandInvoker) error {
    for cq.Size() > 0 {
        cmd, err := cq.Dequeue()
        if err != nil {
            return err
        }

        if err := invoker.ExecuteCommand(cmd); err != nil {
            return fmt.Errorf("failed to execute queued command %s: %w", cmd.GetID(), err)
        }
    }
    return nil
}

// Example usage
func main() {
    fmt.Println("=== Command Pattern Demo ===\n")

    // Create account service and accounts
    accountService := NewBankAccountService()
    accountService.CreateAccount("ACC_001", decimal.NewFromFloat(1000.0))
    accountService.CreateAccount("ACC_002", decimal.NewFromFloat(500.0))
    accountService.CreateAccount("ACC_003", decimal.NewFromFloat(2000.0))

    // Create command invoker
    invoker := NewCommandInvoker()

    // Example 1: Simple commands with undo/redo
    fmt.Println("=== Simple Commands ===")

    creditCmd := NewCreditAccountCommand(accountService, "ACC_001", decimal.NewFromFloat(200.0))
    debitCmd := NewDebitAccountCommand(accountService, "ACC_002", decimal.NewFromFloat(100.0))
    transferCmd := NewTransferMoneyCommand(accountService, "ACC_003", "ACC_001", decimal.NewFromFloat(300.0))

    // Execute commands
    fmt.Println("Executing commands...")
    invoker.ExecuteCommand(creditCmd)
    invoker.ExecuteCommand(debitCmd)
    invoker.ExecuteCommand(transferCmd)

    // Print balances
    printBalances(accountService, []string{"ACC_001", "ACC_002", "ACC_003"})

    // Undo commands
    fmt.Println("\nUndoing commands...")
    invoker.Undo() // Undo transfer
    invoker.Undo() // Undo debit

    printBalances(accountService, []string{"ACC_001", "ACC_002", "ACC_003"})

    // Redo commands
    fmt.Println("\nRedoing commands...")
    invoker.Redo() // Redo debit
    invoker.Redo() // Redo transfer

    printBalances(accountService, []string{"ACC_001", "ACC_002", "ACC_003"})

    // Example 2: Macro command
    fmt.Println("\n=== Macro Command ===")

    // Create a macro command that performs multiple operations atomically
    macroCommands := []Command{
        NewDebitAccountCommand(accountService, "ACC_001", decimal.NewFromFloat(150.0)),
        NewCreditAccountCommand(accountService, "ACC_002", decimal.NewFromFloat(150.0)),
        NewFreezeAccountCommand(accountService, "ACC_003"),
    }

    macroCmd := NewMacroCommand("Multi-step transaction", macroCommands...)

    fmt.Println("Executing macro command...")
    err := invoker.ExecuteCommand(macroCmd)
    if err != nil {
        fmt.Printf("Macro command failed: %v\n", err)
    }

    printBalances(accountService, []string{"ACC_001", "ACC_002", "ACC_003"})

    // Undo macro command
    fmt.Println("\nUndoing macro command...")
    invoker.Undo()

    printBalances(accountService, []string{"ACC_001", "ACC_002", "ACC_003"})

    // Example 3: Command queue
    fmt.Println("\n=== Command Queue ===")

    queue := NewCommandQueue()

    // Add commands to queue
    queue.Enqueue(NewCreditAccountCommand(accountService, "ACC_001", decimal.NewFromFloat(50.0)))
    queue.Enqueue(NewCreditAccountCommand(accountService, "ACC_002", decimal.NewFromFloat(75.0)))
    queue.Enqueue(NewTransferMoneyCommand(accountService, "ACC_001", "ACC_003", decimal.NewFromFloat(100.0)))

    fmt.Printf("Queue size: %d\n", queue.Size())

    // Execute all queued commands
    fmt.Println("Executing queued commands...")
    err = queue.ExecuteAll(invoker)
    if err != nil {
        fmt.Printf("Queue execution failed: %v\n", err)
    }

    printBalances(accountService, []string{"ACC_001", "ACC_002", "ACC_003"})

    // Example 4: Command history
    fmt.Println("\n=== Command History ===")

    history := invoker.GetHistory()
    fmt.Printf("Command history (%d commands):\n", len(history))
    for i, cmd := range history {
        fmt.Printf("%d. %s (ID: %s, Executed: %t)\n",
            i+1, cmd.GetDescription(), cmd.GetID(), cmd.IsExecuted())
    }

    fmt.Printf("\nCan undo: %t\n", invoker.CanUndo())
    fmt.Printf("Can redo: %t\n", invoker.CanRedo())

    fmt.Println("\n=== Command Pattern Demo Complete ===")
}

func printBalances(accountService AccountService, accountIDs []string) {
    fmt.Println("Account balances:")
    for _, accountID := range accountIDs {
        balance, err := accountService.GetBalance(accountID)
        if err != nil {
            fmt.Printf("  %s: Error - %v\n", accountID, err)
        } else {
            fmt.Printf("  %s: %s\n", accountID, balance.String())
        }
    }
}
```

## Variants & Trade-offs

### Variants

1. **Simple Command**

```go
type SimpleCommand struct {
    action func() error
    undo   func() error
}

func (s *SimpleCommand) Execute() error { return s.action() }
func (s *SimpleCommand) Undo() error    { return s.undo() }
```

2. **Asynchronous Command**

```go
type AsyncCommand struct {
    command Command
    result  chan error
}

func (a *AsyncCommand) ExecuteAsync() {
    go func() {
        a.result <- a.command.Execute()
    }()
}

func (a *AsyncCommand) Wait() error {
    return <-a.result
}
```

3. **Transactional Command**

```go
type TransactionalCommand struct {
    commands    []Command
    transaction Transaction
}

func (t *TransactionalCommand) Execute() error {
    tx := t.transaction.Begin()

    for _, cmd := range t.commands {
        if err := cmd.Execute(); err != nil {
            tx.Rollback()
            return err
        }
    }

    return tx.Commit()
}
```

4. **Scheduled Command**

```go
type ScheduledCommand struct {
    command   Command
    executeAt time.Time
    scheduler Scheduler
}

func (s *ScheduledCommand) Schedule() {
    s.scheduler.ScheduleAt(s.executeAt, s.command)
}
```

### Trade-offs

**Pros:**

- **Undo/Redo Support**: Easy to implement undo and redo functionality
- **Decoupling**: Separates invoker from receiver
- **Queuing**: Commands can be queued for later execution
- **Logging**: Commands can be logged for auditing
- **Macro Operations**: Can compose complex operations
- **Transactional**: Can group operations into transactions

**Cons:**

- **Memory Overhead**: Storing command objects requires memory
- **Complexity**: Adds complexity for simple operations
- **Performance**: Extra indirection may impact performance
- **State Management**: Need to manage command state carefully
- **Debugging**: Can be harder to debug complex command chains

**When to Choose Command vs Alternatives:**

| Scenario            | Pattern      | Reason           |
| ------------------- | ------------ | ---------------- |
| Need undo/redo      | Command      | Built-in support |
| Simple method calls | Direct calls | Less overhead    |
| Event handling      | Observer     | Event-driven     |
| Request queuing     | Command      | Natural fit      |
| API operations      | Command      | Encapsulation    |

## Testable Example

```go
package main

import (
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "github.com/stretchr/testify/mock"
    "github.com/shopspring/decimal"
)

// Mock Account Service for testing
type MockAccountService struct {
    mock.Mock
}

func (m *MockAccountService) GetBalance(accountID string) (decimal.Decimal, error) {
    args := m.Called(accountID)
    return args.Get(0).(decimal.Decimal), args.Error(1)
}

func (m *MockAccountService) CreditAccount(accountID string, amount decimal.Decimal, transactionID string) error {
    args := m.Called(accountID, amount, transactionID)
    return args.Error(0)
}

func (m *MockAccountService) DebitAccount(accountID string, amount decimal.Decimal, transactionID string) error {
    args := m.Called(accountID, amount, transactionID)
    return args.Error(0)
}

func (m *MockAccountService) TransferMoney(fromAccount, toAccount string, amount decimal.Decimal, transactionID string) error {
    args := m.Called(fromAccount, toAccount, amount, transactionID)
    return args.Error(0)
}

func (m *MockAccountService) FreezeAccount(accountID string) error {
    args := m.Called(accountID)
    return args.Error(0)
}

func (m *MockAccountService) UnfreezeAccount(accountID string) error {
    args := m.Called(accountID)
    return args.Error(0)
}

func TestCreditAccountCommand_Execute(t *testing.T) {
    mockService := &MockAccountService{}
    amount := decimal.NewFromFloat(100.0)

    cmd := NewCreditAccountCommand(mockService, "ACC_001", amount)

    // Setup expectation
    mockService.On("CreditAccount", "ACC_001", amount, mock.AnythingOfType("string")).Return(nil)

    // Execute command
    err := cmd.Execute()

    // Assert
    assert.NoError(t, err)
    assert.True(t, cmd.IsExecuted())
    mockService.AssertExpectations(t)
}

func TestCreditAccountCommand_Undo(t *testing.T) {
    mockService := &MockAccountService{}
    amount := decimal.NewFromFloat(100.0)

    cmd := NewCreditAccountCommand(mockService, "ACC_001", amount)

    // Setup expectations
    mockService.On("CreditAccount", "ACC_001", amount, mock.AnythingOfType("string")).Return(nil)
    mockService.On("DebitAccount", "ACC_001", amount, mock.AnythingOfType("string")).Return(nil)

    // Execute and then undo
    err := cmd.Execute()
    require.NoError(t, err)

    err = cmd.Undo()
    assert.NoError(t, err)
    mockService.AssertExpectations(t)
}

func TestCreditAccountCommand_UndoWithoutExecute(t *testing.T) {
    mockService := &MockAccountService{}
    amount := decimal.NewFromFloat(100.0)

    cmd := NewCreditAccountCommand(mockService, "ACC_001", amount)

    // Try to undo without executing
    err := cmd.Undo()

    assert.Error(t, err)
    assert.Contains(t, err.Error(), "cannot undo: command not executed")
}

func TestTransferMoneyCommand_Execute(t *testing.T) {
    mockService := &MockAccountService{}
    amount := decimal.NewFromFloat(200.0)

    cmd := NewTransferMoneyCommand(mockService, "ACC_001", "ACC_002", amount)

    // Setup expectation
    mockService.On("TransferMoney", "ACC_001", "ACC_002", amount, mock.AnythingOfType("string")).Return(nil)

    // Execute command
    err := cmd.Execute()

    // Assert
    assert.NoError(t, err)
    assert.True(t, cmd.IsExecuted())
    mockService.AssertExpectations(t)
}

func TestTransferMoneyCommand_Undo(t *testing.T) {
    mockService := &MockAccountService{}
    amount := decimal.NewFromFloat(200.0)

    cmd := NewTransferMoneyCommand(mockService, "ACC_001", "ACC_002", amount)

    // Setup expectations
    mockService.On("TransferMoney", "ACC_001", "ACC_002", amount, mock.AnythingOfType("string")).Return(nil)
    mockService.On("TransferMoney", "ACC_002", "ACC_001", amount, mock.AnythingOfType("string")).Return(nil)

    // Execute and then undo
    err := cmd.Execute()
    require.NoError(t, err)

    err = cmd.Undo()
    assert.NoError(t, err)
    mockService.AssertExpectations(t)
}

func TestMacroCommand_Execute(t *testing.T) {
    mockService := &MockAccountService{}

    // Create individual commands
    creditCmd := NewCreditAccountCommand(mockService, "ACC_001", decimal.NewFromFloat(100.0))
    debitCmd := NewDebitAccountCommand(mockService, "ACC_002", decimal.NewFromFloat(50.0))

    // Create macro command
    macroCmd := NewMacroCommand("Test macro", creditCmd, debitCmd)

    // Setup expectations
    mockService.On("CreditAccount", "ACC_001", decimal.NewFromFloat(100.0), mock.AnythingOfType("string")).Return(nil)
    mockService.On("DebitAccount", "ACC_002", decimal.NewFromFloat(50.0), mock.AnythingOfType("string")).Return(nil)

    // Execute macro command
    err := macroCmd.Execute()

    // Assert
    assert.NoError(t, err)
    assert.True(t, macroCmd.IsExecuted())
    assert.True(t, creditCmd.IsExecuted())
    assert.True(t, debitCmd.IsExecuted())
    mockService.AssertExpectations(t)
}

func TestMacroCommand_ExecuteWithFailure(t *testing.T) {
    mockService := &MockAccountService{}

    // Create individual commands
    creditCmd := NewCreditAccountCommand(mockService, "ACC_001", decimal.NewFromFloat(100.0))
    debitCmd := NewDebitAccountCommand(mockService, "ACC_002", decimal.NewFromFloat(50.0))

    // Create macro command
    macroCmd := NewMacroCommand("Test macro with failure", creditCmd, debitCmd)

    // Setup expectations - first succeeds, second fails
    mockService.On("CreditAccount", "ACC_001", decimal.NewFromFloat(100.0), mock.AnythingOfType("string")).Return(nil)
    mockService.On("DebitAccount", "ACC_002", decimal.NewFromFloat(50.0), mock.AnythingOfType("string")).Return(fmt.Errorf("insufficient funds"))
    mockService.On("DebitAccount", "ACC_001", decimal.NewFromFloat(100.0), mock.AnythingOfType("string")).Return(nil) // Rollback

    // Execute macro command
    err := macroCmd.Execute()

    // Assert
    assert.Error(t, err)
    assert.False(t, macroCmd.IsExecuted())
    assert.Contains(t, err.Error(), "insufficient funds")
    mockService.AssertExpectations(t)
}

func TestMacroCommand_Undo(t *testing.T) {
    mockService := &MockAccountService{}

    // Create individual commands
    creditCmd := NewCreditAccountCommand(mockService, "ACC_001", decimal.NewFromFloat(100.0))
    debitCmd := NewDebitAccountCommand(mockService, "ACC_002", decimal.NewFromFloat(50.0))

    // Create macro command
    macroCmd := NewMacroCommand("Test macro undo", creditCmd, debitCmd)

    // Setup expectations for execute
    mockService.On("CreditAccount", "ACC_001", decimal.NewFromFloat(100.0), mock.AnythingOfType("string")).Return(nil)
    mockService.On("DebitAccount", "ACC_002", decimal.NewFromFloat(50.0), mock.AnythingOfType("string")).Return(nil)

    // Setup expectations for undo (in reverse order)
    mockService.On("CreditAccount", "ACC_002", decimal.NewFromFloat(50.0), mock.AnythingOfType("string")).Return(nil)
    mockService.On("DebitAccount", "ACC_001", decimal.NewFromFloat(100.0), mock.AnythingOfType("string")).Return(nil)

    // Execute and then undo macro command
    err := macroCmd.Execute()
    require.NoError(t, err)

    err = macroCmd.Undo()
    assert.NoError(t, err)
    mockService.AssertExpectations(t)
}

func TestCommandInvoker_ExecuteCommand(t *testing.T) {
    mockService := &MockAccountService{}
    invoker := NewCommandInvoker()

    cmd := NewCreditAccountCommand(mockService, "ACC_001", decimal.NewFromFloat(100.0))

    // Setup expectation
    mockService.On("CreditAccount", "ACC_001", decimal.NewFromFloat(100.0), mock.AnythingOfType("string")).Return(nil)

    // Execute command through invoker
    err := invoker.ExecuteCommand(cmd)

    // Assert
    assert.NoError(t, err)
    assert.Equal(t, 1, invoker.GetHistorySize())
    assert.True(t, invoker.CanUndo())
    assert.False(t, invoker.CanRedo())
    mockService.AssertExpectations(t)
}

func TestCommandInvoker_Undo(t *testing.T) {
    mockService := &MockAccountService{}
    invoker := NewCommandInvoker()

    cmd := NewCreditAccountCommand(mockService, "ACC_001", decimal.NewFromFloat(100.0))

    // Setup expectations
    mockService.On("CreditAccount", "ACC_001", decimal.NewFromFloat(100.0), mock.AnythingOfType("string")).Return(nil)
    mockService.On("DebitAccount", "ACC_001", decimal.NewFromFloat(100.0), mock.AnythingOfType("string")).Return(nil)

    // Execute and undo
    err := invoker.ExecuteCommand(cmd)
    require.NoError(t, err)

    err = invoker.Undo()
    assert.NoError(t, err)
    assert.False(t, invoker.CanUndo())
    assert.True(t, invoker.CanRedo())
    mockService.AssertExpectations(t)
}

func TestCommandInvoker_Redo(t *testing.T) {
    mockService := &MockAccountService{}
    invoker := NewCommandInvoker()

    cmd := NewCreditAccountCommand(mockService, "ACC_001", decimal.NewFromFloat(100.0))

    // Setup expectations (command will be executed twice)
    mockService.On("CreditAccount", "ACC_001", decimal.NewFromFloat(100.0), mock.AnythingOfType("string")).Return(nil).Times(2)
    mockService.On("DebitAccount", "ACC_001", decimal.NewFromFloat(100.0), mock.AnythingOfType("string")).Return(nil)

    // Execute, undo, and redo
    err := invoker.ExecuteCommand(cmd)
    require.NoError(t, err)

    err = invoker.Undo()
    require.NoError(t, err)

    err = invoker.Redo()
    assert.NoError(t, err)
    assert.True(t, invoker.CanUndo())
    assert.False(t, invoker.CanRedo())
    mockService.AssertExpectations(t)
}

func TestCommandInvoker_UndoEmptyHistory(t *testing.T) {
    invoker := NewCommandInvoker()

    err := invoker.Undo()

    assert.Error(t, err)
    assert.Contains(t, err.Error(), "no commands to undo")
}

func TestCommandInvoker_RedoAtEndOfHistory(t *testing.T) {
    mockService := &MockAccountService{}
    invoker := NewCommandInvoker()

    cmd := NewCreditAccountCommand(mockService, "ACC_001", decimal.NewFromFloat(100.0))

    // Setup expectation
    mockService.On("CreditAccount", "ACC_001", decimal.NewFromFloat(100.0), mock.AnythingOfType("string")).Return(nil)

    // Execute command
    err := invoker.ExecuteCommand(cmd)
    require.NoError(t, err)

    // Try to redo when at end of history
    err = invoker.Redo()

    assert.Error(t, err)
    assert.Contains(t, err.Error(), "no commands to redo")
    mockService.AssertExpectations(t)
}

func TestCommandQueue_EnqueueDequeue(t *testing.T) {
    mockService := &MockAccountService{}
    queue := NewCommandQueue()

    cmd1 := NewCreditAccountCommand(mockService, "ACC_001", decimal.NewFromFloat(100.0))
    cmd2 := NewDebitAccountCommand(mockService, "ACC_002", decimal.NewFromFloat(50.0))

    // Enqueue commands
    queue.Enqueue(cmd1)
    queue.Enqueue(cmd2)

    assert.Equal(t, 2, queue.Size())

    // Dequeue commands
    dequeuedCmd1, err := queue.Dequeue()
    assert.NoError(t, err)
    assert.Equal(t, cmd1.GetID(), dequeuedCmd1.GetID())
    assert.Equal(t, 1, queue.Size())

    dequeuedCmd2, err := queue.Dequeue()
    assert.NoError(t, err)
    assert.Equal(t, cmd2.GetID(), dequeuedCmd2.GetID())
    assert.Equal(t, 0, queue.Size())

    // Try to dequeue from empty queue
    _, err = queue.Dequeue()
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "queue is empty")
}

func TestCommandQueue_ExecuteAll(t *testing.T) {
    mockService := &MockAccountService{}
    queue := NewCommandQueue()
    invoker := NewCommandInvoker()

    cmd1 := NewCreditAccountCommand(mockService, "ACC_001", decimal.NewFromFloat(100.0))
    cmd2 := NewDebitAccountCommand(mockService, "ACC_002", decimal.NewFromFloat(50.0))

    // Setup expectations
    mockService.On("CreditAccount", "ACC_001", decimal.NewFromFloat(100.0), mock.AnythingOfType("string")).Return(nil)
    mockService.On("DebitAccount", "ACC_002", decimal.NewFromFloat(50.0), mock.AnythingOfType("string")).Return(nil)

    // Enqueue commands
    queue.Enqueue(cmd1)
    queue.Enqueue(cmd2)

    // Execute all
    err := queue.ExecuteAll(invoker)

    assert.NoError(t, err)
    assert.Equal(t, 0, queue.Size())
    assert.Equal(t, 2, invoker.GetHistorySize())
    mockService.AssertExpectations(t)
}

func BenchmarkCreditAccountCommand_Execute(b *testing.B) {
    accountService := NewBankAccountService()
    accountService.CreateAccount("ACC_001", decimal.NewFromFloat(1000.0))

    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        cmd := NewCreditAccountCommand(accountService, "ACC_001", decimal.NewFromFloat(1.0))
        err := cmd.Execute()
        if err != nil {
            b.Fatal(err)
        }
    }
}

func BenchmarkCommandInvoker_ExecuteCommand(b *testing.B) {
    accountService := NewBankAccountService()
    accountService.CreateAccount("ACC_001", decimal.NewFromFloat(1000.0))
    invoker := NewCommandInvoker()

    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        cmd := NewCreditAccountCommand(accountService, "ACC_001", decimal.NewFromFloat(1.0))
        err := invoker.ExecuteCommand(cmd)
        if err != nil {
            b.Fatal(err)
        }
    }
}
```

## Integration Tips

### 1. Persistence Integration

```go
type PersistentCommandInvoker struct {
    *CommandInvoker
    storage CommandStorage
}

type CommandStorage interface {
    SaveCommand(cmd Command) error
    LoadCommands() ([]Command, error)
    DeleteCommand(id string) error
}

func (p *PersistentCommandInvoker) ExecuteCommand(cmd Command) error {
    if err := p.CommandInvoker.ExecuteCommand(cmd); err != nil {
        return err
    }

    // Persist command
    return p.storage.SaveCommand(cmd)
}

func (p *PersistentCommandInvoker) LoadHistory() error {
    commands, err := p.storage.LoadCommands()
    if err != nil {
        return err
    }

    p.history = commands
    p.currentPos = len(commands) - 1
    return nil
}
```

### 2. Event-Driven Integration

```go
type EventDrivenCommand struct {
    BaseCommand
    eventBus EventBus
}

func (e *EventDrivenCommand) Execute() error {
    err := e.executeInternal()

    if err == nil {
        e.eventBus.Publish(CommandExecutedEvent{
            CommandID: e.GetID(),
            Timestamp: time.Now(),
        })
    } else {
        e.eventBus.Publish(CommandFailedEvent{
            CommandID: e.GetID(),
            Error:     err.Error(),
            Timestamp: time.Now(),
        })
    }

    return err
}
```

### 3. Circuit Breaker Integration

```go
type ResilientCommand struct {
    command        Command
    circuitBreaker *CircuitBreaker
}

func (r *ResilientCommand) Execute() error {
    return r.circuitBreaker.Execute(func() error {
        return r.command.Execute()
    })
}
```

### 4. Metrics Integration

```go
type MetricsCommand struct {
    command Command
    metrics MetricsCollector
}

func (m *MetricsCommand) Execute() error {
    start := time.Now()
    err := m.command.Execute()
    duration := time.Since(start)

    labels := map[string]string{
        "command_type": reflect.TypeOf(m.command).Name(),
        "success":      fmt.Sprintf("%t", err == nil),
    }

    m.metrics.RecordDuration("command_execution_time", duration, labels)
    m.metrics.IncrementCounter("command_executions", labels)

    return err
}
```

## Common Interview Questions

### 1. **How does Command pattern enable undo/redo functionality?**

**Answer:**
Command pattern enables undo/redo by:

1. **Encapsulating Operations**: Each operation is wrapped in a command object
2. **Storing State**: Commands store the information needed to reverse operations
3. **Command History**: Invoker maintains a history of executed commands
4. **Undo Implementation**: Each command implements an undo method that reverses its effect

```go
type TransferCommand struct {
    fromAccount string
    toAccount   string
    amount      decimal.Decimal
    executed    bool
}

func (t *TransferCommand) Execute() error {
    err := transferMoney(t.fromAccount, t.toAccount, t.amount)
    if err == nil {
        t.executed = true
    }
    return err
}

func (t *TransferCommand) Undo() error {
    if !t.executed {
        return fmt.Errorf("cannot undo: not executed")
    }
    // Reverse the transfer
    return transferMoney(t.toAccount, t.fromAccount, t.amount)
}

// Command history enables undo/redo
type CommandInvoker struct {
    history    []Command
    currentPos int
}

func (ci *CommandInvoker) Undo() error {
    if ci.currentPos < 0 {
        return fmt.Errorf("nothing to undo")
    }

    cmd := ci.history[ci.currentPos]
    err := cmd.Undo()
    if err == nil {
        ci.currentPos--
    }
    return err
}

func (ci *CommandInvoker) Redo() error {
    if ci.currentPos >= len(ci.history)-1 {
        return fmt.Errorf("nothing to redo")
    }

    ci.currentPos++
    return ci.history[ci.currentPos].Execute()
}
```

### 2. **What's the difference between Command and Strategy patterns?**

**Answer:**
| Aspect | Command | Strategy |
|--------|---------|----------|
| **Purpose** | Encapsulate requests as objects | Encapsulate algorithms |
| **When Used** | For actions, operations, transactions | For algorithms, calculations |
| **Undo Support** | Natural support for undo/redo | No undo concept |
| **State** | Commands often store state | Strategies are typically stateless |
| **Lifecycle** | Commands may be stored/queued | Strategies are usually short-lived |

**Example:**

```go
// Command - encapsulates an action with state
type PaymentCommand struct {
    amount      decimal.Decimal
    fromAccount string
    toAccount   string
    executed    bool // State for undo
}

func (p *PaymentCommand) Execute() error {
    // Perform payment and store state
    p.executed = true
    return processPayment(p.fromAccount, p.toAccount, p.amount)
}

func (p *PaymentCommand) Undo() error {
    // Use stored state to undo
    return reversePayment(p.toAccount, p.fromAccount, p.amount)
}

// Strategy - encapsulates algorithm without state
type PaymentStrategy interface {
    CalculateFee(amount decimal.Decimal) decimal.Decimal
}

type CreditCardStrategy struct{}
func (c *CreditCardStrategy) CalculateFee(amount decimal.Decimal) decimal.Decimal {
    return amount.Mul(decimal.NewFromFloat(0.029)) // 2.9%
}

type WireTransferStrategy struct{}
func (w *WireTransferStrategy) CalculateFee(amount decimal.Decimal) decimal.Decimal {
    return decimal.NewFromFloat(25.00) // Fixed fee
}
```

### 3. **How do you implement macro commands (composite commands)?**

**Answer:**
Macro commands combine multiple commands into a single operation:

```go
type MacroCommand struct {
    commands []Command
    executed bool
}

func (m *MacroCommand) Execute() error {
    for i, cmd := range m.commands {
        if err := cmd.Execute(); err != nil {
            // Rollback previously executed commands
            for j := i - 1; j >= 0; j-- {
                if rollbackErr := m.commands[j].Undo(); rollbackErr != nil {
                    log.Printf("Rollback failed: %v", rollbackErr)
                }
            }
            return err
        }
    }

    m.executed = true
    return nil
}

func (m *MacroCommand) Undo() error {
    if !m.executed {
        return fmt.Errorf("macro command not executed")
    }

    // Undo commands in reverse order
    for i := len(m.commands) - 1; i >= 0; i-- {
        if err := m.commands[i].Undo(); err != nil {
            return err
        }
    }

    return nil
}

// Usage
func CreateAccountTransfer(from, to string, amount decimal.Decimal) *MacroCommand {
    return &MacroCommand{
        commands: []Command{
            &DebitCommand{Account: from, Amount: amount},
            &CreditCommand{Account: to, Amount: amount},
            &LogCommand{Message: fmt.Sprintf("Transferred %s from %s to %s", amount, from, to)},
        },
    }
}
```

**Key features:**

- **Atomicity**: All commands succeed or all are rolled back
- **Order**: Commands execute in order, undo in reverse order
- **Composition**: Macro commands can contain other macro commands
- **Transaction-like behavior**: Either all operations complete or none do

### 4. **How do you handle command failures and error recovery?**

**Answer:**
Handle command failures through several strategies:

1. **Try-Catch with Rollback**:

```go
func (c *ComplexCommand) Execute() error {
    // Track what was done for rollback
    var completedSteps []func() error

    defer func() {
        if err := recover(); err != nil {
            // Rollback completed steps
            for i := len(completedSteps) - 1; i >= 0; i-- {
                completedSteps[i]()
            }
        }
    }()

    // Step 1
    if err := c.step1(); err != nil {
        return err
    }
    completedSteps = append(completedSteps, c.undoStep1)

    // Step 2
    if err := c.step2(); err != nil {
        return err
    }
    completedSteps = append(completedSteps, c.undoStep2)

    return nil
}
```

2. **Compensating Actions**:

```go
type CompensatingCommand struct {
    mainAction   func() error
    compensation func() error
    executed     bool
}

func (c *CompensatingCommand) Execute() error {
    err := c.mainAction()
    if err == nil {
        c.executed = true
    }
    return err
}

func (c *CompensatingCommand) Undo() error {
    if !c.executed {
        return nil
    }
    return c.compensation()
}
```

3. **Retry Logic**:

```go
type RetryableCommand struct {
    command    Command
    maxRetries int
    backoff    time.Duration
}

func (r *RetryableCommand) Execute() error {
    var lastErr error

    for attempt := 0; attempt <= r.maxRetries; attempt++ {
        err := r.command.Execute()
        if err == nil {
            return nil
        }

        lastErr = err
        if attempt < r.maxRetries {
            time.Sleep(r.backoff * time.Duration(attempt+1))
        }
    }

    return fmt.Errorf("command failed after %d attempts: %w", r.maxRetries+1, lastErr)
}
```

### 5. **How do you implement command queuing and scheduling?**

**Answer:**
Implement command queuing and scheduling through several mechanisms:

1. **Simple Queue**:

```go
type CommandQueue struct {
    commands chan Command
    workers  int
}

func NewCommandQueue(workers int) *CommandQueue {
    return &CommandQueue{
        commands: make(chan Command, 100),
        workers:  workers,
    }
}

func (q *CommandQueue) Start() {
    for i := 0; i < q.workers; i++ {
        go q.worker()
    }
}

func (q *CommandQueue) worker() {
    for cmd := range q.commands {
        if err := cmd.Execute(); err != nil {
            log.Printf("Command execution failed: %v", err)
        }
    }
}

func (q *CommandQueue) Enqueue(cmd Command) {
    q.commands <- cmd
}
```

2. **Priority Queue**:

```go
type PriorityCommand struct {
    Command
    Priority int
}

type PriorityQueue []*PriorityCommand

func (pq *PriorityQueue) Push(x interface{}) {
    *pq = append(*pq, x.(*PriorityCommand))
}

func (pq *PriorityQueue) Pop() interface{} {
    old := *pq
    n := len(old)
    item := old[n-1]
    *pq = old[0 : n-1]
    return item
}
```

3. **Scheduled Execution**:

```go
type ScheduledCommand struct {
    Command
    ExecuteAt time.Time
}

type CommandScheduler struct {
    commands []ScheduledCommand
    ticker   *time.Ticker
}

func (s *CommandScheduler) Schedule(cmd Command, executeAt time.Time) {
    s.commands = append(s.commands, ScheduledCommand{
        Command:   cmd,
        ExecuteAt: executeAt,
    })
}

func (s *CommandScheduler) Start() {
    s.ticker = time.NewTicker(time.Second)
    go func() {
        for range s.ticker.C {
            s.executeReadyCommands()
        }
    }()
}

func (s *CommandScheduler) executeReadyCommands() {
    now := time.Now()
    var remaining []ScheduledCommand

    for _, scheduledCmd := range s.commands {
        if scheduledCmd.ExecuteAt.Before(now) || scheduledCmd.ExecuteAt.Equal(now) {
            go scheduledCmd.Execute()
        } else {
            remaining = append(remaining, scheduledCmd)
        }
    }

    s.commands = remaining
}
```
