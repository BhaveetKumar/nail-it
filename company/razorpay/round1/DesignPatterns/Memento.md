# Memento Pattern

## Pattern Name & Intent

**Memento** is a behavioral design pattern that provides the ability to save and restore the previous state of an object without violating encapsulation. It allows capturing an object's internal state in a memento object and restoring it later.

**Key Intent:**

- Capture and externalize an object's internal state
- Restore object to a previous state (undo functionality)
- Preserve encapsulation boundaries
- Support multiple checkpoints and rollback operations
- Enable state history management
- Implement undo/redo mechanisms

## When to Use

**Use Memento when:**

1. **Undo/Redo Operations**: Need to implement undo functionality
2. **Checkpointing**: Creating snapshots of object state at specific points
3. **State Recovery**: Ability to restore to previous valid state after failures
4. **Transaction Management**: Rolling back changes if transaction fails
5. **Configuration Management**: Saving and restoring application settings
6. **Game Development**: Save/load game states
7. **Document Editing**: Version control and history management

**Don't use when:**

- Object state is simple and doesn't change frequently
- Memory usage is a critical concern (mementos can be memory-intensive)
- State is too large to copy efficiently
- Real-time systems where memento creation/restoration is too slow

## Real-World Use Cases (Payments/Fintech)

### 1. Payment Transaction State Management

```go
// Payment state memento for transaction rollback
type PaymentMemento struct {
    transactionID   string
    status          string
    amount          decimal.Decimal
    balances        map[string]decimal.Decimal
    timestamp       time.Time
    attemptCount    int
    gatewayResponse map[string]interface{}
    riskScore       float64
    metadata        map[string]interface{}
}

type PaymentTransaction struct {
    id              string
    status          string
    amount          decimal.Decimal
    balances        map[string]decimal.Decimal
    attemptCount    int
    gatewayResponse map[string]interface{}
    riskScore       float64
    metadata        map[string]interface{}
    history         []*PaymentMemento
    logger          *zap.Logger
}

func (pt *PaymentTransaction) CreateMemento() *PaymentMemento {
    // Deep copy balances
    balancesCopy := make(map[string]decimal.Decimal)
    for k, v := range pt.balances {
        balancesCopy[k] = v
    }

    // Deep copy metadata
    metadataCopy := make(map[string]interface{})
    for k, v := range pt.metadata {
        metadataCopy[k] = v
    }

    // Deep copy gateway response
    gatewayResponseCopy := make(map[string]interface{})
    for k, v := range pt.gatewayResponse {
        gatewayResponseCopy[k] = v
    }

    memento := &PaymentMemento{
        transactionID:   pt.id,
        status:          pt.status,
        amount:          pt.amount,
        balances:        balancesCopy,
        timestamp:       time.Now(),
        attemptCount:    pt.attemptCount,
        gatewayResponse: gatewayResponseCopy,
        riskScore:       pt.riskScore,
        metadata:        metadataCopy,
    }

    // Store in history
    pt.history = append(pt.history, memento)

    pt.logger.Debug("Payment memento created",
        zap.String("transaction_id", pt.id),
        zap.String("status", pt.status),
        zap.Int("history_count", len(pt.history)))

    return memento
}

func (pt *PaymentTransaction) RestoreFromMemento(memento *PaymentMemento) error {
    if memento.transactionID != pt.id {
        return fmt.Errorf("memento transaction ID mismatch: expected %s, got %s",
            pt.id, memento.transactionID)
    }

    pt.logger.Info("Restoring payment transaction from memento",
        zap.String("transaction_id", pt.id),
        zap.String("current_status", pt.status),
        zap.String("restore_status", memento.status),
        zap.Time("memento_timestamp", memento.timestamp))

    // Restore state
    pt.status = memento.status
    pt.amount = memento.amount
    pt.attemptCount = memento.attemptCount
    pt.riskScore = memento.riskScore

    // Deep copy balances back
    pt.balances = make(map[string]decimal.Decimal)
    for k, v := range memento.balances {
        pt.balances[k] = v
    }

    // Deep copy metadata back
    pt.metadata = make(map[string]interface{})
    for k, v := range memento.metadata {
        pt.metadata[k] = v
    }

    // Deep copy gateway response back
    pt.gatewayResponse = make(map[string]interface{})
    for k, v := range memento.gatewayResponse {
        pt.gatewayResponse[k] = v
    }

    return nil
}

func (pt *PaymentTransaction) GetHistory() []*PaymentMemento {
    history := make([]*PaymentMemento, len(pt.history))
    copy(history, pt.history)
    return history
}

func (pt *PaymentTransaction) RollbackToStatus(targetStatus string) error {
    // Find the most recent memento with the target status
    for i := len(pt.history) - 1; i >= 0; i-- {
        memento := pt.history[i]
        if memento.status == targetStatus {
            pt.logger.Info("Rolling back to previous status",
                zap.String("transaction_id", pt.id),
                zap.String("current_status", pt.status),
                zap.String("target_status", targetStatus),
                zap.Time("memento_timestamp", memento.timestamp))

            return pt.RestoreFromMemento(memento)
        }
    }

    return fmt.Errorf("no memento found with status %s", targetStatus)
}

func (pt *PaymentTransaction) RollbackToTimestamp(targetTime time.Time) error {
    // Find the memento closest to but not after the target time
    var closestMemento *PaymentMemento
    var closestDiff time.Duration = time.Duration(math.MaxInt64)

    for _, memento := range pt.history {
        if memento.timestamp.Before(targetTime) || memento.timestamp.Equal(targetTime) {
            diff := targetTime.Sub(memento.timestamp)
            if diff < closestDiff {
                closestDiff = diff
                closestMemento = memento
            }
        }
    }

    if closestMemento == nil {
        return fmt.Errorf("no memento found before or at timestamp %s", targetTime.Format(time.RFC3339))
    }

    pt.logger.Info("Rolling back to timestamp",
        zap.String("transaction_id", pt.id),
        zap.Time("target_time", targetTime),
        zap.Time("memento_time", closestMemento.timestamp))

    return pt.RestoreFromMemento(closestMemento)
}

// Payment processor with memento support
type PaymentProcessor struct {
    transaction *PaymentTransaction
    gateway     PaymentGateway
    validator   PaymentValidator
    logger      *zap.Logger
}

func (pp *PaymentProcessor) ProcessPayment(request *PaymentRequest) (*PaymentResult, error) {
    // Create checkpoint before processing
    checkpoint := pp.transaction.CreateMemento()

    pp.logger.Info("Processing payment with checkpoint",
        zap.String("transaction_id", pp.transaction.id),
        zap.String("amount", request.Amount.String()))

    // Update transaction state
    pp.transaction.status = "PROCESSING"
    pp.transaction.attemptCount++

    // Validate payment
    if err := pp.validator.ValidatePayment(request); err != nil {
        pp.logger.Warn("Payment validation failed, rolling back",
            zap.String("transaction_id", pp.transaction.id),
            zap.Error(err))

        // Rollback to checkpoint
        if rollbackErr := pp.transaction.RestoreFromMemento(checkpoint); rollbackErr != nil {
            pp.logger.Error("Failed to rollback transaction", zap.Error(rollbackErr))
        }

        return nil, fmt.Errorf("validation failed: %w", err)
    }

    // Process with gateway
    result, err := pp.gateway.ProcessPayment(request)
    if err != nil {
        pp.logger.Warn("Gateway processing failed, rolling back",
            zap.String("transaction_id", pp.transaction.id),
            zap.Error(err))

        // Rollback to checkpoint
        if rollbackErr := pp.transaction.RestoreFromMemento(checkpoint); rollbackErr != nil {
            pp.logger.Error("Failed to rollback transaction", zap.Error(rollbackErr))
        }

        return nil, fmt.Errorf("gateway processing failed: %w", err)
    }

    // Update transaction with successful result
    pp.transaction.status = "COMPLETED"
    pp.transaction.gatewayResponse = map[string]interface{}{
        "transaction_id": result.TransactionID,
        "gateway_status": result.Status,
        "processed_at":   result.ProcessedAt,
    }

    // Create success checkpoint
    pp.transaction.CreateMemento()

    pp.logger.Info("Payment processed successfully",
        zap.String("transaction_id", pp.transaction.id),
        zap.String("gateway_transaction_id", result.TransactionID))

    return result, nil
}
```

### 2. Account Balance State Management

```go
// Account balance memento for financial operations
type BalanceMemento struct {
    accountID       string
    balance         decimal.Decimal
    lockedAmount    decimal.Decimal
    transactions    []string
    lastUpdated     time.Time
    version         int64
    metadata        map[string]interface{}
}

type Account struct {
    id              string
    balance         decimal.Decimal
    lockedAmount    decimal.Decimal
    transactions    []string
    lastUpdated     time.Time
    version         int64
    metadata        map[string]interface{}
    snapshots       []*BalanceMemento
    maxSnapshots    int
    logger          *zap.Logger
    mu              sync.RWMutex
}

func (a *Account) CreateSnapshot() *BalanceMemento {
    a.mu.RLock()
    defer a.mu.RUnlock()

    // Deep copy transactions
    transactionsCopy := make([]string, len(a.transactions))
    copy(transactionsCopy, a.transactions)

    // Deep copy metadata
    metadataCopy := make(map[string]interface{})
    for k, v := range a.metadata {
        metadataCopy[k] = v
    }

    snapshot := &BalanceMemento{
        accountID:    a.id,
        balance:      a.balance,
        lockedAmount: a.lockedAmount,
        transactions: transactionsCopy,
        lastUpdated:  time.Now(),
        version:      a.version,
        metadata:     metadataCopy,
    }

    a.snapshots = append(a.snapshots, snapshot)

    // Maintain maximum snapshots
    if len(a.snapshots) > a.maxSnapshots {
        a.snapshots = a.snapshots[1:]
    }

    a.logger.Debug("Account snapshot created",
        zap.String("account_id", a.id),
        zap.String("balance", a.balance.String()),
        zap.Int("snapshot_count", len(a.snapshots)))

    return snapshot
}

func (a *Account) RestoreFromSnapshot(snapshot *BalanceMemento) error {
    a.mu.Lock()
    defer a.mu.Unlock()

    if snapshot.accountID != a.id {
        return fmt.Errorf("snapshot account ID mismatch: expected %s, got %s",
            a.id, snapshot.accountID)
    }

    a.logger.Info("Restoring account from snapshot",
        zap.String("account_id", a.id),
        zap.String("current_balance", a.balance.String()),
        zap.String("snapshot_balance", snapshot.balance.String()),
        zap.Time("snapshot_time", snapshot.lastUpdated))

    // Restore state
    a.balance = snapshot.balance
    a.lockedAmount = snapshot.lockedAmount
    a.version = snapshot.version
    a.lastUpdated = snapshot.lastUpdated

    // Deep copy transactions back
    a.transactions = make([]string, len(snapshot.transactions))
    copy(a.transactions, snapshot.transactions)

    // Deep copy metadata back
    a.metadata = make(map[string]interface{})
    for k, v := range snapshot.metadata {
        a.metadata[k] = v
    }

    return nil
}

func (a *Account) GetSnapshots() []*BalanceMemento {
    a.mu.RLock()
    defer a.mu.RUnlock()

    snapshots := make([]*BalanceMemento, len(a.snapshots))
    copy(snapshots, a.snapshots)
    return snapshots
}

func (a *Account) RollbackToVersion(targetVersion int64) error {
    a.mu.Lock()
    defer a.mu.Unlock()

    // Find snapshot with target version
    for i := len(a.snapshots) - 1; i >= 0; i-- {
        snapshot := a.snapshots[i]
        if snapshot.version == targetVersion {
            return a.restoreFromSnapshotUnsafe(snapshot)
        }
    }

    return fmt.Errorf("no snapshot found with version %d", targetVersion)
}

func (a *Account) restoreFromSnapshotUnsafe(snapshot *BalanceMemento) error {
    // Restore state without locking (called from locked methods)
    a.balance = snapshot.balance
    a.lockedAmount = snapshot.lockedAmount
    a.version = snapshot.version
    a.lastUpdated = snapshot.lastUpdated

    a.transactions = make([]string, len(snapshot.transactions))
    copy(a.transactions, snapshot.transactions)

    a.metadata = make(map[string]interface{})
    for k, v := range snapshot.metadata {
        a.metadata[k] = v
    }

    return nil
}

// Balance manager with transaction rollback
type BalanceManager struct {
    account *Account
    logger  *zap.Logger
}

func (bm *BalanceManager) Transfer(amount decimal.Decimal, toAccountID string) error {
    // Create snapshot before transfer
    snapshot := bm.account.CreateSnapshot()

    bm.logger.Info("Starting transfer with snapshot",
        zap.String("from_account", bm.account.id),
        zap.String("to_account", toAccountID),
        zap.String("amount", amount.String()))

    // Check available balance
    if bm.account.balance.LessThan(amount) {
        return fmt.Errorf("insufficient balance: have %s, need %s",
            bm.account.balance.String(), amount.String())
    }

    // Lock the amount first
    bm.account.mu.Lock()
    bm.account.lockedAmount = bm.account.lockedAmount.Add(amount)
    bm.account.version++
    bm.account.mu.Unlock()

    // Simulate transfer processing (could fail)
    if err := bm.processTransfer(amount, toAccountID); err != nil {
        bm.logger.Warn("Transfer failed, rolling back",
            zap.String("account_id", bm.account.id),
            zap.Error(err))

        // Rollback to snapshot
        if rollbackErr := bm.account.RestoreFromSnapshot(snapshot); rollbackErr != nil {
            bm.logger.Error("Failed to rollback account", zap.Error(rollbackErr))
            return fmt.Errorf("transfer failed and rollback failed: %w", rollbackErr)
        }

        return fmt.Errorf("transfer failed: %w", err)
    }

    // Complete the transfer
    bm.account.mu.Lock()
    bm.account.balance = bm.account.balance.Sub(amount)
    bm.account.lockedAmount = bm.account.lockedAmount.Sub(amount)
    bm.account.transactions = append(bm.account.transactions,
        fmt.Sprintf("TRANSFER_%s_%s", amount.String(), toAccountID))
    bm.account.version++
    bm.account.lastUpdated = time.Now()
    bm.account.mu.Unlock()

    // Create success snapshot
    bm.account.CreateSnapshot()

    bm.logger.Info("Transfer completed successfully",
        zap.String("account_id", bm.account.id),
        zap.String("new_balance", bm.account.balance.String()))

    return nil
}

func (bm *BalanceManager) processTransfer(amount decimal.Decimal, toAccountID string) error {
    // Simulate external transfer processing
    // This could involve API calls, database updates, etc.

    // Simulate random failures for demonstration
    if rand.Float32() < 0.1 { // 10% failure rate
        return fmt.Errorf("external transfer service failed")
    }

    // Simulate processing time
    time.Sleep(100 * time.Millisecond)

    return nil
}
```

### 3. Trading Strategy State Management

```go
// Trading strategy memento for risk management
type StrategyMemento struct {
    strategyID      string
    positions       map[string]decimal.Decimal
    cashBalance     decimal.Decimal
    riskLimits      map[string]decimal.Decimal
    parameters      map[string]interface{}
    performance     PerformanceMetrics
    timestamp       time.Time
    marketData      map[string]decimal.Decimal
}

type PerformanceMetrics struct {
    TotalReturn    decimal.Decimal
    Volatility     decimal.Decimal
    MaxDrawdown    decimal.Decimal
    SharpeRatio    decimal.Decimal
    TradeCount     int
}

type TradingStrategy struct {
    id          string
    positions   map[string]decimal.Decimal
    cashBalance decimal.Decimal
    riskLimits  map[string]decimal.Decimal
    parameters  map[string]interface{}
    performance PerformanceMetrics
    marketData  map[string]decimal.Decimal
    checkpoints []*StrategyMemento
    logger      *zap.Logger
    mu          sync.RWMutex
}

func (ts *TradingStrategy) CreateCheckpoint() *StrategyMemento {
    ts.mu.RLock()
    defer ts.mu.RUnlock()

    // Deep copy positions
    positionsCopy := make(map[string]decimal.Decimal)
    for k, v := range ts.positions {
        positionsCopy[k] = v
    }

    // Deep copy risk limits
    riskLimitsCopy := make(map[string]decimal.Decimal)
    for k, v := range ts.riskLimits {
        riskLimitsCopy[k] = v
    }

    // Deep copy parameters
    parametersCopy := make(map[string]interface{})
    for k, v := range ts.parameters {
        parametersCopy[k] = v
    }

    // Deep copy market data
    marketDataCopy := make(map[string]decimal.Decimal)
    for k, v := range ts.marketData {
        marketDataCopy[k] = v
    }

    checkpoint := &StrategyMemento{
        strategyID:  ts.id,
        positions:   positionsCopy,
        cashBalance: ts.cashBalance,
        riskLimits:  riskLimitsCopy,
        parameters:  parametersCopy,
        performance: ts.performance, // struct copy
        timestamp:   time.Now(),
        marketData:  marketDataCopy,
    }

    ts.checkpoints = append(ts.checkpoints, checkpoint)

    ts.logger.Debug("Strategy checkpoint created",
        zap.String("strategy_id", ts.id),
        zap.String("cash_balance", ts.cashBalance.String()),
        zap.Int("position_count", len(ts.positions)),
        zap.Int("checkpoint_count", len(ts.checkpoints)))

    return checkpoint
}

func (ts *TradingStrategy) RestoreFromCheckpoint(checkpoint *StrategyMemento) error {
    ts.mu.Lock()
    defer ts.mu.Unlock()

    if checkpoint.strategyID != ts.id {
        return fmt.Errorf("checkpoint strategy ID mismatch: expected %s, got %s",
            ts.id, checkpoint.strategyID)
    }

    ts.logger.Info("Restoring strategy from checkpoint",
        zap.String("strategy_id", ts.id),
        zap.String("current_cash", ts.cashBalance.String()),
        zap.String("checkpoint_cash", checkpoint.cashBalance.String()),
        zap.Time("checkpoint_time", checkpoint.timestamp))

    // Restore state
    ts.cashBalance = checkpoint.cashBalance
    ts.performance = checkpoint.performance

    // Deep copy positions back
    ts.positions = make(map[string]decimal.Decimal)
    for k, v := range checkpoint.positions {
        ts.positions[k] = v
    }

    // Deep copy risk limits back
    ts.riskLimits = make(map[string]decimal.Decimal)
    for k, v := range checkpoint.riskLimits {
        ts.riskLimits[k] = v
    }

    // Deep copy parameters back
    ts.parameters = make(map[string]interface{})
    for k, v := range checkpoint.parameters {
        ts.parameters[k] = v
    }

    // Deep copy market data back
    ts.marketData = make(map[string]decimal.Decimal)
    for k, v := range checkpoint.marketData {
        ts.marketData[k] = v
    }

    return nil
}

func (ts *TradingStrategy) ExecuteTrade(symbol string, quantity decimal.Decimal, price decimal.Decimal) error {
    // Create checkpoint before trade
    checkpoint := ts.CreateCheckpoint()

    ts.logger.Info("Executing trade with checkpoint",
        zap.String("strategy_id", ts.id),
        zap.String("symbol", symbol),
        zap.String("quantity", quantity.String()),
        zap.String("price", price.String()))

    // Calculate trade cost
    tradeCost := quantity.Abs().Mul(price)

    // Check cash availability for buy orders
    if quantity.IsPositive() && ts.cashBalance.LessThan(tradeCost) {
        return fmt.Errorf("insufficient cash: have %s, need %s",
            ts.cashBalance.String(), tradeCost.String())
    }

    // Check risk limits
    if err := ts.checkRiskLimits(symbol, quantity, price); err != nil {
        ts.logger.Warn("Risk limit violation, not executing trade",
            zap.String("symbol", symbol),
            zap.Error(err))
        return fmt.Errorf("risk limit violation: %w", err)
    }

    // Simulate trade execution (could fail)
    if err := ts.executeTrade(symbol, quantity, price); err != nil {
        ts.logger.Warn("Trade execution failed, rolling back",
            zap.String("strategy_id", ts.id),
            zap.String("symbol", symbol),
            zap.Error(err))

        // Rollback to checkpoint
        if rollbackErr := ts.RestoreFromCheckpoint(checkpoint); rollbackErr != nil {
            ts.logger.Error("Failed to rollback strategy", zap.Error(rollbackErr))
            return fmt.Errorf("trade failed and rollback failed: %w", rollbackErr)
        }

        return fmt.Errorf("trade execution failed: %w", err)
    }

    // Update strategy state
    ts.mu.Lock()
    currentPosition := ts.positions[symbol]
    ts.positions[symbol] = currentPosition.Add(quantity)

    if quantity.IsPositive() {
        // Buy order - reduce cash
        ts.cashBalance = ts.cashBalance.Sub(tradeCost)
    } else {
        // Sell order - increase cash
        ts.cashBalance = ts.cashBalance.Add(tradeCost)
    }

    // Update performance metrics
    ts.updatePerformanceMetrics(symbol, quantity, price)
    ts.mu.Unlock()

    // Create success checkpoint
    ts.CreateCheckpoint()

    ts.logger.Info("Trade executed successfully",
        zap.String("strategy_id", ts.id),
        zap.String("symbol", symbol),
        zap.String("new_position", ts.positions[symbol].String()),
        zap.String("new_cash_balance", ts.cashBalance.String()))

    return nil
}

func (ts *TradingStrategy) checkRiskLimits(symbol string, quantity decimal.Decimal, price decimal.Decimal) error {
    // Check position size limit
    if positionLimit, exists := ts.riskLimits["max_position_size"]; exists {
        currentPosition := ts.positions[symbol]
        newPosition := currentPosition.Add(quantity)
        if newPosition.Abs().GreaterThan(positionLimit) {
            return fmt.Errorf("position size limit exceeded: %s > %s",
                newPosition.Abs().String(), positionLimit.String())
        }
    }

    // Check single trade size limit
    if tradeLimit, exists := ts.riskLimits["max_trade_size"]; exists {
        tradeValue := quantity.Abs().Mul(price)
        if tradeValue.GreaterThan(tradeLimit) {
            return fmt.Errorf("trade size limit exceeded: %s > %s",
                tradeValue.String(), tradeLimit.String())
        }
    }

    return nil
}

func (ts *TradingStrategy) executeTrade(symbol string, quantity decimal.Decimal, price decimal.Decimal) error {
    // Simulate external trade execution
    // This could involve broker API calls, market data validation, etc.

    // Simulate random failures
    if rand.Float32() < 0.05 { // 5% failure rate
        return fmt.Errorf("trade execution failed: market conditions")
    }

    // Simulate execution time
    time.Sleep(50 * time.Millisecond)

    return nil
}

func (ts *TradingStrategy) updatePerformanceMetrics(symbol string, quantity decimal.Decimal, price decimal.Decimal) {
    // Update trade count
    ts.performance.TradeCount++

    // Calculate P&L (simplified)
    tradeValue := quantity.Mul(price)
    ts.performance.TotalReturn = ts.performance.TotalReturn.Add(tradeValue)

    // Other performance calculations would go here
    // Volatility, Sharpe ratio, max drawdown, etc.
}

func (ts *TradingStrategy) RollbackToTimestamp(targetTime time.Time) error {
    ts.mu.Lock()
    defer ts.mu.Unlock()

    // Find the checkpoint closest to but not after the target time
    var closestCheckpoint *StrategyMemento
    var closestDiff time.Duration = time.Duration(math.MaxInt64)

    for _, checkpoint := range ts.checkpoints {
        if checkpoint.timestamp.Before(targetTime) || checkpoint.timestamp.Equal(targetTime) {
            diff := targetTime.Sub(checkpoint.timestamp)
            if diff < closestDiff {
                closestDiff = diff
                closestCheckpoint = checkpoint
            }
        }
    }

    if closestCheckpoint == nil {
        return fmt.Errorf("no checkpoint found before or at timestamp %s",
            targetTime.Format(time.RFC3339))
    }

    ts.logger.Info("Rolling back strategy to timestamp",
        zap.String("strategy_id", ts.id),
        zap.Time("target_time", targetTime),
        zap.Time("checkpoint_time", closestCheckpoint.timestamp))

    return ts.restoreFromCheckpointUnsafe(closestCheckpoint)
}

func (ts *TradingStrategy) restoreFromCheckpointUnsafe(checkpoint *StrategyMemento) error {
    // Restore without locking (called from locked methods)
    ts.cashBalance = checkpoint.cashBalance
    ts.performance = checkpoint.performance

    ts.positions = make(map[string]decimal.Decimal)
    for k, v := range checkpoint.positions {
        ts.positions[k] = v
    }

    ts.riskLimits = make(map[string]decimal.Decimal)
    for k, v := range checkpoint.riskLimits {
        ts.riskLimits[k] = v
    }

    ts.parameters = make(map[string]interface{})
    for k, v := range checkpoint.parameters {
        ts.parameters[k] = v
    }

    ts.marketData = make(map[string]decimal.Decimal)
    for k, v := range checkpoint.marketData {
        ts.marketData[k] = v
    }

    return nil
}
```

## Go Implementation

```go
package main

import (
    "fmt"
    "time"
    "strings"
    "go.uber.org/zap"
)

// Memento interface
type Memento interface {
    GetState() interface{}
    GetTimestamp() time.Time
    GetVersion() int
}

// Originator interface
type Originator interface {
    CreateMemento() Memento
    RestoreFromMemento(memento Memento) error
    GetCurrentState() interface{}
}

// Caretaker interface
type Caretaker interface {
    SaveMemento(memento Memento) error
    GetMemento(version int) (Memento, error)
    GetMementos() []Memento
    GetMementoCount() int
}

// Example: Text editor with undo/redo functionality
// This demonstrates the memento pattern for document state management

// Document state
type DocumentState struct {
    Content    string
    CursorPos  int
    Selection  Selection
    Formatting map[int]TextFormat
    Metadata   map[string]interface{}
}

type Selection struct {
    Start int
    End   int
}

type TextFormat struct {
    Bold      bool
    Italic    bool
    Underline bool
    FontSize  int
    Color     string
}

// Document memento
type DocumentMemento struct {
    state     *DocumentState
    timestamp time.Time
    version   int
    operation string
}

func (dm *DocumentMemento) GetState() interface{} {
    return dm.state
}

func (dm *DocumentMemento) GetTimestamp() time.Time {
    return dm.timestamp
}

func (dm *DocumentMemento) GetVersion() int {
    return dm.version
}

func (dm *DocumentMemento) GetOperation() string {
    return dm.operation
}

// Document (Originator)
type TextDocument struct {
    state         *DocumentState
    version       int
    logger        *zap.Logger
}

func NewTextDocument(logger *zap.Logger) *TextDocument {
    return &TextDocument{
        state: &DocumentState{
            Content:    "",
            CursorPos:  0,
            Selection:  Selection{Start: 0, End: 0},
            Formatting: make(map[int]TextFormat),
            Metadata:   make(map[string]interface{}),
        },
        version: 0,
        logger:  logger,
    }
}

func (td *TextDocument) CreateMemento() Memento {
    td.version++

    // Deep copy the state
    stateCopy := &DocumentState{
        Content:   td.state.Content,
        CursorPos: td.state.CursorPos,
        Selection: Selection{
            Start: td.state.Selection.Start,
            End:   td.state.Selection.End,
        },
        Formatting: make(map[int]TextFormat),
        Metadata:   make(map[string]interface{}),
    }

    // Deep copy formatting
    for k, v := range td.state.Formatting {
        stateCopy.Formatting[k] = v
    }

    // Deep copy metadata
    for k, v := range td.state.Metadata {
        stateCopy.Metadata[k] = v
    }

    memento := &DocumentMemento{
        state:     stateCopy,
        timestamp: time.Now(),
        version:   td.version,
        operation: "manual_save",
    }

    td.logger.Debug("Document memento created",
        zap.Int("version", td.version),
        zap.Int("content_length", len(td.state.Content)),
        zap.Int("cursor_pos", td.state.CursorPos))

    return memento
}

func (td *TextDocument) CreateMementoWithOperation(operation string) Memento {
    memento := td.CreateMemento().(*DocumentMemento)
    memento.operation = operation
    return memento
}

func (td *TextDocument) RestoreFromMemento(memento Memento) error {
    docMemento, ok := memento.(*DocumentMemento)
    if !ok {
        return fmt.Errorf("invalid memento type")
    }

    state, ok := docMemento.GetState().(*DocumentState)
    if !ok {
        return fmt.Errorf("invalid state type in memento")
    }

    td.logger.Info("Restoring document from memento",
        zap.Int("current_version", td.version),
        zap.Int("restore_version", docMemento.GetVersion()),
        zap.String("operation", docMemento.GetOperation()),
        zap.Time("memento_time", docMemento.GetTimestamp()))

    // Deep copy the state back
    td.state = &DocumentState{
        Content:   state.Content,
        CursorPos: state.CursorPos,
        Selection: Selection{
            Start: state.Selection.Start,
            End:   state.Selection.End,
        },
        Formatting: make(map[int]TextFormat),
        Metadata:   make(map[string]interface{}),
    }

    // Deep copy formatting back
    for k, v := range state.Formatting {
        td.state.Formatting[k] = v
    }

    // Deep copy metadata back
    for k, v := range state.Metadata {
        td.state.Metadata[k] = v
    }

    td.version = docMemento.GetVersion()

    return nil
}

func (td *TextDocument) GetCurrentState() interface{} {
    return td.state
}

// Document operations
func (td *TextDocument) InsertText(text string, position int) {
    if position < 0 || position > len(td.state.Content) {
        position = len(td.state.Content)
    }

    td.state.Content = td.state.Content[:position] + text + td.state.Content[position:]
    td.state.CursorPos = position + len(text)

    td.logger.Debug("Text inserted",
        zap.String("text", text),
        zap.Int("position", position),
        zap.Int("new_cursor_pos", td.state.CursorPos))
}

func (td *TextDocument) DeleteText(start, end int) string {
    if start < 0 {
        start = 0
    }
    if end > len(td.state.Content) {
        end = len(td.state.Content)
    }
    if start > end {
        start, end = end, start
    }

    deleted := td.state.Content[start:end]
    td.state.Content = td.state.Content[:start] + td.state.Content[end:]
    td.state.CursorPos = start

    td.logger.Debug("Text deleted",
        zap.String("deleted_text", deleted),
        zap.Int("start", start),
        zap.Int("end", end),
        zap.Int("new_cursor_pos", td.state.CursorPos))

    return deleted
}

func (td *TextDocument) SetCursorPosition(position int) {
    if position < 0 {
        position = 0
    }
    if position > len(td.state.Content) {
        position = len(td.state.Content)
    }

    td.state.CursorPos = position
}

func (td *TextDocument) SetSelection(start, end int) {
    if start < 0 {
        start = 0
    }
    if end > len(td.state.Content) {
        end = len(td.state.Content)
    }
    if start > end {
        start, end = end, start
    }

    td.state.Selection = Selection{Start: start, End: end}
}

func (td *TextDocument) ApplyFormatting(start, end int, format TextFormat) {
    for i := start; i < end; i++ {
        td.state.Formatting[i] = format
    }
}

func (td *TextDocument) GetContent() string {
    return td.state.Content
}

func (td *TextDocument) GetCursorPosition() int {
    return td.state.CursorPos
}

func (td *TextDocument) GetSelection() Selection {
    return td.state.Selection
}

func (td *TextDocument) GetWordCount() int {
    return len(strings.Fields(td.state.Content))
}

func (td *TextDocument) GetCharacterCount() int {
    return len(td.state.Content)
}

// Document history caretaker
type DocumentHistory struct {
    mementos    []Memento
    maxHistory  int
    currentPos  int
    logger      *zap.Logger
}

func NewDocumentHistory(maxHistory int, logger *zap.Logger) *DocumentHistory {
    return &DocumentHistory{
        mementos:   make([]Memento, 0),
        maxHistory: maxHistory,
        currentPos: -1,
        logger:     logger,
    }
}

func (dh *DocumentHistory) SaveMemento(memento Memento) error {
    // If we're not at the end of history, remove future mementos
    if dh.currentPos < len(dh.mementos)-1 {
        dh.mementos = dh.mementos[:dh.currentPos+1]
    }

    // Add new memento
    dh.mementos = append(dh.mementos, memento)
    dh.currentPos = len(dh.mementos) - 1

    // Maintain maximum history size
    if len(dh.mementos) > dh.maxHistory {
        dh.mementos = dh.mementos[1:]
        dh.currentPos--
    }

    dh.logger.Debug("Memento saved to history",
        zap.Int("version", memento.GetVersion()),
        zap.Int("history_size", len(dh.mementos)),
        zap.Int("current_pos", dh.currentPos))

    return nil
}

func (dh *DocumentHistory) GetMemento(version int) (Memento, error) {
    for _, memento := range dh.mementos {
        if memento.GetVersion() == version {
            return memento, nil
        }
    }
    return nil, fmt.Errorf("memento with version %d not found", version)
}

func (dh *DocumentHistory) GetMementos() []Memento {
    mementos := make([]Memento, len(dh.mementos))
    copy(mementos, dh.mementos)
    return mementos
}

func (dh *DocumentHistory) GetMementoCount() int {
    return len(dh.mementos)
}

func (dh *DocumentHistory) CanUndo() bool {
    return dh.currentPos > 0
}

func (dh *DocumentHistory) CanRedo() bool {
    return dh.currentPos < len(dh.mementos)-1
}

func (dh *DocumentHistory) GetUndoMemento() (Memento, error) {
    if !dh.CanUndo() {
        return nil, fmt.Errorf("cannot undo: no previous state")
    }

    dh.currentPos--
    memento := dh.mementos[dh.currentPos]

    dh.logger.Debug("Undo memento retrieved",
        zap.Int("version", memento.GetVersion()),
        zap.Int("current_pos", dh.currentPos))

    return memento, nil
}

func (dh *DocumentHistory) GetRedoMemento() (Memento, error) {
    if !dh.CanRedo() {
        return nil, fmt.Errorf("cannot redo: no future state")
    }

    dh.currentPos++
    memento := dh.mementos[dh.currentPos]

    dh.logger.Debug("Redo memento retrieved",
        zap.Int("version", memento.GetVersion()),
        zap.Int("current_pos", dh.currentPos))

    return memento, nil
}

func (dh *DocumentHistory) GetCurrentMemento() (Memento, error) {
    if dh.currentPos < 0 || dh.currentPos >= len(dh.mementos) {
        return nil, fmt.Errorf("no current memento")
    }

    return dh.mementos[dh.currentPos], nil
}

func (dh *DocumentHistory) GetHistoryInfo() string {
    if len(dh.mementos) == 0 {
        return "No history available"
    }

    var info strings.Builder
    info.WriteString(fmt.Sprintf("History: %d entries, current position: %d\n",
        len(dh.mementos), dh.currentPos))

    for i, memento := range dh.mementos {
        prefix := "  "
        if i == dh.currentPos {
            prefix = "* "
        }

        if docMemento, ok := memento.(*DocumentMemento); ok {
            info.WriteString(fmt.Sprintf("%s%d: %s (%s)\n",
                prefix,
                memento.GetVersion(),
                docMemento.GetOperation(),
                memento.GetTimestamp().Format("15:04:05")))
        } else {
            info.WriteString(fmt.Sprintf("%s%d: version %d (%s)\n",
                prefix,
                i,
                memento.GetVersion(),
                memento.GetTimestamp().Format("15:04:05")))
        }
    }

    return info.String()
}

// Text editor with undo/redo
type TextEditor struct {
    document *TextDocument
    history  *DocumentHistory
    logger   *zap.Logger
}

func NewTextEditor(logger *zap.Logger) *TextEditor {
    document := NewTextDocument(logger)
    history := NewDocumentHistory(50, logger) // Keep last 50 states

    // Save initial state
    initialMemento := document.CreateMementoWithOperation("initial_state")
    history.SaveMemento(initialMemento)

    return &TextEditor{
        document: document,
        history:  history,
        logger:   logger,
    }
}

func (te *TextEditor) Type(text string) {
    // Save state before modification
    memento := te.document.CreateMementoWithOperation(fmt.Sprintf("type_%s", text))
    te.history.SaveMemento(memento)

    // Perform the operation
    te.document.InsertText(text, te.document.GetCursorPosition())

    te.logger.Info("Text typed",
        zap.String("text", text),
        zap.Int("cursor_pos", te.document.GetCursorPosition()))
}

func (te *TextEditor) Delete() {
    if te.document.GetCursorPosition() == 0 {
        return
    }

    // Save state before modification
    memento := te.document.CreateMementoWithOperation("delete")
    te.history.SaveMemento(memento)

    // Delete character before cursor
    pos := te.document.GetCursorPosition()
    deleted := te.document.DeleteText(pos-1, pos)

    te.logger.Info("Character deleted",
        zap.String("deleted_char", deleted),
        zap.Int("cursor_pos", te.document.GetCursorPosition()))
}

func (te *TextEditor) DeleteSelection() {
    selection := te.document.GetSelection()
    if selection.Start == selection.End {
        return
    }

    // Save state before modification
    memento := te.document.CreateMementoWithOperation("delete_selection")
    te.history.SaveMemento(memento)

    // Delete selected text
    deleted := te.document.DeleteText(selection.Start, selection.End)

    te.logger.Info("Selection deleted",
        zap.String("deleted_text", deleted),
        zap.Int("start", selection.Start),
        zap.Int("end", selection.End))
}

func (te *TextEditor) Undo() error {
    if !te.history.CanUndo() {
        return fmt.Errorf("cannot undo: no previous state")
    }

    memento, err := te.history.GetUndoMemento()
    if err != nil {
        return err
    }

    if err := te.document.RestoreFromMemento(memento); err != nil {
        return fmt.Errorf("failed to restore from memento: %w", err)
    }

    te.logger.Info("Undo performed",
        zap.Int("restored_version", memento.GetVersion()))

    return nil
}

func (te *TextEditor) Redo() error {
    if !te.history.CanRedo() {
        return fmt.Errorf("cannot redo: no future state")
    }

    memento, err := te.history.GetRedoMemento()
    if err != nil {
        return err
    }

    if err := te.document.RestoreFromMemento(memento); err != nil {
        return fmt.Errorf("failed to restore from memento: %w", err)
    }

    te.logger.Info("Redo performed",
        zap.Int("restored_version", memento.GetVersion()))

    return nil
}

func (te *TextEditor) SetCursor(position int) {
    te.document.SetCursorPosition(position)
}

func (te *TextEditor) Select(start, end int) {
    te.document.SetSelection(start, end)
}

func (te *TextEditor) ApplyBold() {
    selection := te.document.GetSelection()
    if selection.Start == selection.End {
        return
    }

    // Save state before modification
    memento := te.document.CreateMementoWithOperation("apply_bold")
    te.history.SaveMemento(memento)

    format := TextFormat{Bold: true, FontSize: 12, Color: "black"}
    te.document.ApplyFormatting(selection.Start, selection.End, format)

    te.logger.Info("Bold formatting applied",
        zap.Int("start", selection.Start),
        zap.Int("end", selection.End))
}

func (te *TextEditor) GetContent() string {
    return te.document.GetContent()
}

func (te *TextEditor) GetStats() (int, int) {
    return te.document.GetWordCount(), te.document.GetCharacterCount()
}

func (te *TextEditor) GetHistoryInfo() string {
    return te.history.GetHistoryInfo()
}

func (te *TextEditor) CanUndo() bool {
    return te.history.CanUndo()
}

func (te *TextEditor) CanRedo() bool {
    return te.history.CanRedo()
}

// Example usage
func main() {
    fmt.Println("=== Memento Pattern Demo ===\n")

    // Create logger
    logger, _ := zap.NewDevelopment()
    defer logger.Sync()

    // Create text editor
    editor := NewTextEditor(logger)

    fmt.Println("=== Initial State ===")
    fmt.Printf("Content: '%s'\n", editor.GetContent())
    words, chars := editor.GetStats()
    fmt.Printf("Words: %d, Characters: %d\n", words, chars)
    fmt.Printf("Can Undo: %t, Can Redo: %t\n\n", editor.CanUndo(), editor.CanRedo())

    // Type some text
    fmt.Println("=== Typing Text ===")
    editor.Type("Hello")
    editor.Type(" ")
    editor.Type("World")
    editor.Type("!")

    fmt.Printf("Content: '%s'\n", editor.GetContent())
    words, chars = editor.GetStats()
    fmt.Printf("Words: %d, Characters: %d\n", words, chars)
    fmt.Printf("Can Undo: %t, Can Redo: %t\n\n", editor.CanUndo(), editor.CanRedo())

    // Show history
    fmt.Println("=== Current History ===")
    fmt.Println(editor.GetHistoryInfo())

    // Delete some text
    fmt.Println("=== Deleting Characters ===")
    editor.SetCursor(12) // After "Hello World!"
    editor.Delete()      // Delete "!"
    editor.Delete()      // Delete "d"

    fmt.Printf("Content: '%s'\n", editor.GetContent())
    words, chars = editor.GetStats()
    fmt.Printf("Words: %d, Characters: %d\n", words, chars)
    fmt.Printf("Can Undo: %t, Can Redo: %t\n\n", editor.CanUndo(), editor.CanRedo())

    // Undo operations
    fmt.Println("=== Undo Operations ===")
    for i := 0; i < 3 && editor.CanUndo(); i++ {
        if err := editor.Undo(); err != nil {
            fmt.Printf("Undo failed: %v\n", err)
            break
        }
        fmt.Printf("After undo %d: '%s'\n", i+1, editor.GetContent())
    }

    fmt.Printf("Can Undo: %t, Can Redo: %t\n\n", editor.CanUndo(), editor.CanRedo())

    // Redo operations
    fmt.Println("=== Redo Operations ===")
    for i := 0; i < 2 && editor.CanRedo(); i++ {
        if err := editor.Redo(); err != nil {
            fmt.Printf("Redo failed: %v\n", err)
            break
        }
        fmt.Printf("After redo %d: '%s'\n", i+1, editor.GetContent())
    }

    fmt.Printf("Can Undo: %t, Can Redo: %t\n\n", editor.CanUndo(), editor.CanRedo())

    // Apply formatting
    fmt.Println("=== Applying Formatting ===")
    editor.Select(0, 5) // Select "Hello"
    editor.ApplyBold()

    fmt.Printf("Content: '%s'\n", editor.GetContent())
    fmt.Printf("Applied bold formatting to 'Hello'\n\n")

    // Show final history
    fmt.Println("=== Final History ===")
    fmt.Println(editor.GetHistoryInfo())

    // Test selection deletion
    fmt.Println("=== Selection Deletion ===")
    editor.Select(6, 10) // Select "Worl"
    editor.DeleteSelection()

    fmt.Printf("Content after selection deletion: '%s'\n", editor.GetContent())

    // Undo selection deletion
    fmt.Println("=== Undo Selection Deletion ===")
    if err := editor.Undo(); err != nil {
        fmt.Printf("Undo failed: %v\n", err)
    } else {
        fmt.Printf("Content after undo: '%s'\n", editor.GetContent())
    }

    fmt.Println("\n=== Memento Pattern Demo Complete ===")
}
```

## Variants & Trade-offs

### Variants

1. **Snapshot-based Memento**

```go
// Full state snapshot
type SnapshotMemento struct {
    fullState interface{}
    timestamp time.Time
}

// Advantage: Simple implementation
// Disadvantage: Memory intensive for large objects
```

2. **Incremental Memento**

```go
// Store only changes (delta)
type IncrementalMemento struct {
    changes   []StateChange
    baseState *Memento
    timestamp time.Time
}

type StateChange struct {
    Field     string
    OldValue  interface{}
    NewValue  interface{}
}

// Advantage: Memory efficient
// Disadvantage: Complex restoration logic
```

3. **Compressed Memento**

```go
type CompressedMemento struct {
    compressedData []byte
    compression    string // "gzip", "lz4", etc.
    originalSize   int64
    timestamp      time.Time
}

// Advantage: Reduced memory usage
// Disadvantage: CPU overhead for compression/decompression
```

### Trade-offs

**Pros:**

- **Encapsulation**: Object state is preserved without exposing internal structure
- **Undo/Redo**: Easy implementation of undo/redo functionality
- **State Recovery**: Ability to restore to previous valid states
- **Debugging**: Historical state information helps with debugging
- **Transaction Support**: Can implement transactional behavior

**Cons:**

- **Memory Usage**: Storing multiple object states can be memory-intensive
- **Performance**: Creating and restoring mementos can be slow for large objects
- **Complexity**: Managing memento lifecycle and cleanup can be complex
- **Storage Overhead**: Need to manage when to create and delete mementos
- **Deep Copy Costs**: Creating mementos often requires expensive deep copying

## Integration Tips

### 1. Command Pattern Integration

```go
type UndoableCommand interface {
    Execute() error
    Undo() error
    CreateMemento() Memento
}

type EditCommand struct {
    document   *TextDocument
    operation  func() error
    memento    Memento
}

func (ec *EditCommand) Execute() error {
    // Save state before execution
    ec.memento = ec.document.CreateMemento()
    return ec.operation()
}

func (ec *EditCommand) Undo() error {
    if ec.memento == nil {
        return fmt.Errorf("no memento available for undo")
    }
    return ec.document.RestoreFromMemento(ec.memento)
}
```

### 2. Observer Pattern Integration

```go
type MementoObserver interface {
    OnMementoCreated(memento Memento)
    OnMementoRestored(memento Memento)
}

type ObservableOriginator struct {
    observers []MementoObserver
}

func (oo *ObservableOriginator) AddObserver(observer MementoObserver) {
    oo.observers = append(oo.observers, observer)
}

func (oo *ObservableOriginator) CreateMemento() Memento {
    memento := oo.createMementoInternal()

    // Notify observers
    for _, observer := range oo.observers {
        observer.OnMementoCreated(memento)
    }

    return memento
}
```

### 3. Strategy Pattern Integration

```go
type MementoStrategy interface {
    CreateMemento(originator Originator) Memento
    RestoreMemento(originator Originator, memento Memento) error
}

type FullStateStrategy struct{}
type IncrementalStrategy struct{}
type CompressedStrategy struct{}

type ConfigurableOriginator struct {
    strategy MementoStrategy
}

func (co *ConfigurableOriginator) SetStrategy(strategy MementoStrategy) {
    co.strategy = strategy
}

func (co *ConfigurableOriginator) CreateMemento() Memento {
    return co.strategy.CreateMemento(co)
}
```

## Common Interview Questions

### 1. **How does Memento pattern differ from Command pattern for undo functionality?**

**Answer:**

| Aspect             | Memento Pattern                          | Command Pattern                      |
| ------------------ | ---------------------------------------- | ------------------------------------ |
| **Approach**       | Stores object states                     | Stores operations and their inverses |
| **Undo Mechanism** | Restore previous state                   | Execute inverse operation            |
| **Memory Usage**   | Stores full/partial state snapshots      | Stores operation parameters          |
| **Granularity**    | Object-level state restoration           | Operation-level undo                 |
| **Complexity**     | Simple restore, complex state management | Complex inverse operations           |

**Memento Example:**

```go
// Stores complete document state
type DocumentMemento struct {
    content   string
    cursorPos int
    selection Selection
}

func (d *Document) Undo() {
    // Restore entire previous state
    d.RestoreFromMemento(d.lastMemento)
}
```

**Command Example:**

```go
// Stores operation details
type InsertTextCommand struct {
    document *Document
    text     string
    position int
}

func (itc *InsertTextCommand) Undo() {
    // Execute inverse operation
    itc.document.DeleteText(itc.position, itc.position + len(itc.text))
}
```

### 2. **How do you handle memory management with mementos?**

**Answer:**

**1. Limit History Size:**

```go
type MementoCaretaker struct {
    mementos   []Memento
    maxSize    int
    currentPos int
}

func (mc *MementoCaretaker) SaveMemento(memento Memento) {
    // Remove oldest if at capacity
    if len(mc.mementos) >= mc.maxSize {
        mc.mementos = mc.mementos[1:]
        mc.currentPos--
    }

    mc.mementos = append(mc.mementos, memento)
    mc.currentPos = len(mc.mementos) - 1
}
```

**2. Time-based Cleanup:**

```go
func (mc *MementoCaretaker) CleanupOldMementos(maxAge time.Duration) {
    cutoff := time.Now().Add(-maxAge)

    for i := 0; i < len(mc.mementos); i++ {
        if mc.mementos[i].GetTimestamp().Before(cutoff) {
            mc.mementos = mc.mementos[i+1:]
            mc.currentPos -= (i + 1)
            break
        }
    }
}
```

**3. Lazy Deletion:**

```go
type WeakMemento struct {
    data      *interface{}
    timestamp time.Time
    isValid   bool
}

func (wm *WeakMemento) GetState() interface{} {
    if !wm.isValid || wm.data == nil {
        return nil
    }
    return *wm.data
}

func (wm *WeakMemento) Invalidate() {
    wm.data = nil
    wm.isValid = false
}
```

**4. Compression:**

```go
func (m *Memento) Compress() error {
    data, err := json.Marshal(m.state)
    if err != nil {
        return err
    }

    var buf bytes.Buffer
    writer := gzip.NewWriter(&buf)
    _, err = writer.Write(data)
    writer.Close()

    if err == nil {
        m.compressedData = buf.Bytes()
        m.state = nil // Free original data
        m.compressed = true
    }

    return err
}
```

### 3. **When should you use Memento vs other state management patterns?**

**Answer:**

**Use Memento when:**

- Need complete state restoration
- Undo/redo functionality is required
- State transitions are complex
- Object encapsulation must be preserved
- Rollback to arbitrary previous states is needed

**Use State Machine when:**

- State transitions follow predefined rules
- Current state determines available operations
- State behavior differs significantly
- Need to validate state transitions

**Use Command when:**

- Operations are more important than states
- Need to queue, log, or schedule operations
- Inverse operations are well-defined
- Need macro/batch operation support

**Use Repository/DAO when:**

- State persistence to external storage
- Multiple data sources
- Complex queries on historical data
- Shared state across application instances

**Decision Framework:**

```go
type StateManagementDecision struct {
    NeedUndo        bool
    StateComplexity string // "simple", "medium", "complex"
    StateSize       string // "small", "medium", "large"
    MemoryLimits    bool
    Persistence     bool
    SharedState     bool
}

func (smd *StateManagementDecision) RecommendPattern() string {
    if smd.SharedState && smd.Persistence {
        return "Repository/Event Sourcing"
    }

    if smd.NeedUndo && smd.StateSize == "small" {
        return "Memento"
    }

    if smd.NeedUndo && smd.StateSize == "large" {
        return "Command (with inverse operations)"
    }

    if smd.StateComplexity == "complex" && !smd.NeedUndo {
        return "State Machine"
    }

    return "Simple state management"
}
```

### 4. **How do you implement efficient incremental mementos?**

**Answer:**

**1. Delta-based Mementos:**

```go
type DeltaMemento struct {
    baseMementoID string
    changes       []FieldChange
    timestamp     time.Time
}

type FieldChange struct {
    FieldPath string      // "user.profile.name"
    OldValue  interface{}
    NewValue  interface{}
}

func (d *Document) CreateIncrementalMemento(baseMemento *Memento) *DeltaMemento {
    changes := d.calculateChanges(baseMemento)

    return &DeltaMemento{
        baseMementoID: baseMemento.GetID(),
        changes:       changes,
        timestamp:     time.Now(),
    }
}

func (d *Document) calculateChanges(baseMemento *Memento) []FieldChange {
    var changes []FieldChange
    baseState := baseMemento.GetState().(*DocumentState)

    if d.state.Content != baseState.Content {
        changes = append(changes, FieldChange{
            FieldPath: "content",
            OldValue:  baseState.Content,
            NewValue:  d.state.Content,
        })
    }

    if d.state.CursorPos != baseState.CursorPos {
        changes = append(changes, FieldChange{
            FieldPath: "cursorPos",
            OldValue:  baseState.CursorPos,
            NewValue:  d.state.CursorPos,
        })
    }

    return changes
}
```

**2. Copy-on-Write Mementos:**

```go
type COWMemento struct {
    sharedState *DocumentState
    localCopy   *DocumentState
    modified    map[string]bool
    timestamp   time.Time
}

func (cowm *COWMemento) GetState() interface{} {
    if cowm.localCopy != nil {
        return cowm.localCopy
    }
    return cowm.sharedState
}

func (cowm *COWMemento) ModifyField(fieldName string, newValue interface{}) {
    if cowm.localCopy == nil {
        // Create local copy on first modification
        cowm.localCopy = cowm.deepCopyState(cowm.sharedState)
    }

    cowm.setField(fieldName, newValue)
    cowm.modified[fieldName] = true
}
```

**3. Structural Sharing:**

```go
type SharedMemento struct {
    immutableParts map[string]interface{}
    mutableParts   map[string]interface{}
    version        int64
    timestamp      time.Time
}

func (sm *SharedMemento) ShareImmutableParts(other *SharedMemento) {
    for key, value := range other.immutableParts {
        if _, exists := sm.mutableParts[key]; !exists {
            sm.immutableParts[key] = value
        }
    }
}
```

### 5. **How do you handle concurrent access to mementos?**

**Answer:**

**1. Thread-Safe Caretaker:**

```go
type ThreadSafeCaretaker struct {
    mementos []Memento
    mu       sync.RWMutex
}

func (tsc *ThreadSafeCaretaker) SaveMemento(memento Memento) {
    tsc.mu.Lock()
    defer tsc.mu.Unlock()

    tsc.mementos = append(tsc.mementos, memento)
}

func (tsc *ThreadSafeCaretaker) GetMemento(id string) (Memento, error) {
    tsc.mu.RLock()
    defer tsc.mu.RUnlock()

    for _, memento := range tsc.mementos {
        if memento.GetID() == id {
            return memento, nil
        }
    }

    return nil, fmt.Errorf("memento not found")
}
```

**2. Copy-on-Access:**

```go
func (tsc *ThreadSafeCaretaker) GetMementoSafe(id string) (Memento, error) {
    tsc.mu.RLock()
    original, err := tsc.getMementoUnsafe(id)
    tsc.mu.RUnlock()

    if err != nil {
        return nil, err
    }

    // Return a copy to prevent concurrent modifications
    return original.DeepCopy(), nil
}
```

**3. Immutable Mementos:**

```go
type ImmutableMemento struct {
    state     interface{} // Immutable data structure
    timestamp time.Time
    id        string
}

// Since the memento is immutable, it's inherently thread-safe
func (im *ImmutableMemento) GetState() interface{} {
    return im.state // Safe to return directly
}
```

**4. Channel-based Access:**

```go
type ChannelBasedCaretaker struct {
    saveChan chan SaveRequest
    getChan  chan GetRequest
    done     chan struct{}
}

type SaveRequest struct {
    memento  Memento
    response chan error
}

type GetRequest struct {
    id       string
    response chan GetResponse
}

type GetResponse struct {
    memento Memento
    error   error
}

func (cbc *ChannelBasedCaretaker) worker() {
    mementos := make(map[string]Memento)

    for {
        select {
        case req := <-cbc.saveChan:
            mementos[req.memento.GetID()] = req.memento
            req.response <- nil

        case req := <-cbc.getChan:
            if memento, exists := mementos[req.id]; exists {
                req.response <- GetResponse{memento: memento, error: nil}
            } else {
                req.response <- GetResponse{error: fmt.Errorf("not found")}
            }

        case <-cbc.done:
            return
        }
    }
}
```
