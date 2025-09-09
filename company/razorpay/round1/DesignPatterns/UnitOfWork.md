# Unit of Work Pattern

## Pattern Name & Intent

**Unit of Work** is an architectural pattern that maintains a list of objects affected by a business transaction and coordinates writing out changes and resolving concurrency problems.

**Key Intent:**

- Track changes to objects during a business transaction
- Coordinate writing out changes to the database
- Maintain transactional consistency across multiple operations
- Optimize database access by batching operations
- Handle concurrency conflicts and rollback scenarios
- Provide a clear boundary for business transactions

## When to Use

**Use Unit of Work when:**

1. **Multiple Operations**: Business transactions involve multiple database operations
2. **Transactional Consistency**: Need ACID properties across operations
3. **Performance Optimization**: Want to batch database operations
4. **Change Tracking**: Need to track what objects have been modified
5. **Rollback Scenarios**: Need to undo changes on failure
6. **Concurrency Control**: Managing concurrent access to shared data
7. **Domain Logic**: Complex business rules spanning multiple entities

**Don't use when:**

- Simple CRUD operations with single entities
- Stateless operations that don't require transactions
- High-performance scenarios where overhead is not justified
- Simple applications with minimal business logic

## Real-World Use Cases (Payments/Fintech)

### 1. Payment Processing Unit of Work

```go
// Payment processing involves multiple entities that must be updated atomically
type PaymentUnitOfWork struct {
    // Change tracking
    newPayments      []*PaymentEntity
    modifiedPayments []*PaymentEntity
    deletedPayments  []*PaymentEntity

    newAccounts      []*AccountEntity
    modifiedAccounts []*AccountEntity
    deletedAccounts  []*AccountEntity

    newTransactions      []*TransactionEntity
    modifiedTransactions []*TransactionEntity
    deletedTransactions  []*TransactionEntity

    // Repositories
    paymentRepo     PaymentRepository
    accountRepo     AccountRepository
    transactionRepo TransactionRepository

    // Database transaction
    dbTx   DatabaseTransaction
    logger *zap.Logger

    // State management
    isCommitted bool
    isRolledBack bool
    mu         sync.RWMutex
}

func NewPaymentUnitOfWork(
    paymentRepo PaymentRepository,
    accountRepo AccountRepository,
    transactionRepo TransactionRepository,
    db Database,
    logger *zap.Logger,
) (*PaymentUnitOfWork, error) {
    dbTx, err := db.BeginTransaction()
    if err != nil {
        return nil, fmt.Errorf("failed to begin transaction: %w", err)
    }

    return &PaymentUnitOfWork{
        newPayments:      make([]*PaymentEntity, 0),
        modifiedPayments: make([]*PaymentEntity, 0),
        deletedPayments:  make([]*PaymentEntity, 0),
        newAccounts:      make([]*AccountEntity, 0),
        modifiedAccounts: make([]*AccountEntity, 0),
        deletedAccounts:  make([]*AccountEntity, 0),
        newTransactions:      make([]*TransactionEntity, 0),
        modifiedTransactions: make([]*TransactionEntity, 0),
        deletedTransactions:  make([]*TransactionEntity, 0),
        paymentRepo:     paymentRepo,
        accountRepo:     accountRepo,
        transactionRepo: transactionRepo,
        dbTx:           dbTx,
        logger:         logger,
    }, nil
}

// Register new entities
func (puow *PaymentUnitOfWork) RegisterNewPayment(payment *PaymentEntity) {
    puow.mu.Lock()
    defer puow.mu.Unlock()

    if puow.isCommitted || puow.isRolledBack {
        puow.logger.Warn("Attempting to register entity on completed unit of work")
        return
    }

    puow.newPayments = append(puow.newPayments, payment)
    puow.logger.Debug("Registered new payment",
        zap.String("payment_id", payment.ID),
        zap.String("amount", payment.Amount.String()))
}

func (puow *PaymentUnitOfWork) RegisterNewAccount(account *AccountEntity) {
    puow.mu.Lock()
    defer puow.mu.Unlock()

    if puow.isCommitted || puow.isRolledBack {
        puow.logger.Warn("Attempting to register entity on completed unit of work")
        return
    }

    puow.newAccounts = append(puow.newAccounts, account)
    puow.logger.Debug("Registered new account",
        zap.String("account_id", account.ID),
        zap.String("customer_id", account.CustomerID))
}

func (puow *PaymentUnitOfWork) RegisterNewTransaction(transaction *TransactionEntity) {
    puow.mu.Lock()
    defer puow.mu.Unlock()

    if puow.isCommitted || puow.isRolledBack {
        puow.logger.Warn("Attempting to register entity on completed unit of work")
        return
    }

    puow.newTransactions = append(puow.newTransactions, transaction)
    puow.logger.Debug("Registered new transaction",
        zap.String("transaction_id", transaction.ID),
        zap.String("type", transaction.Type))
}

// Register modified entities
func (puow *PaymentUnitOfWork) RegisterModifiedPayment(payment *PaymentEntity) {
    puow.mu.Lock()
    defer puow.mu.Unlock()

    if puow.isCommitted || puow.isRolledBack {
        puow.logger.Warn("Attempting to register entity on completed unit of work")
        return
    }

    // Check if already in new or modified list
    if !puow.isPaymentInList(payment, puow.newPayments) &&
       !puow.isPaymentInList(payment, puow.modifiedPayments) {
        puow.modifiedPayments = append(puow.modifiedPayments, payment)
        puow.logger.Debug("Registered modified payment",
            zap.String("payment_id", payment.ID))
    }
}

func (puow *PaymentUnitOfWork) RegisterModifiedAccount(account *AccountEntity) {
    puow.mu.Lock()
    defer puow.mu.Unlock()

    if puow.isCommitted || puow.isRolledBack {
        puow.logger.Warn("Attempting to register entity on completed unit of work")
        return
    }

    if !puow.isAccountInList(account, puow.newAccounts) &&
       !puow.isAccountInList(account, puow.modifiedAccounts) {
        puow.modifiedAccounts = append(puow.modifiedAccounts, account)
        puow.logger.Debug("Registered modified account",
            zap.String("account_id", account.ID))
    }
}

// Register deleted entities
func (puow *PaymentUnitOfWork) RegisterDeletedPayment(payment *PaymentEntity) {
    puow.mu.Lock()
    defer puow.mu.Unlock()

    if puow.isCommitted || puow.isRolledBack {
        puow.logger.Warn("Attempting to register entity on completed unit of work")
        return
    }

    // Remove from new/modified if present
    puow.removePaymentFromList(payment, &puow.newPayments)
    puow.removePaymentFromList(payment, &puow.modifiedPayments)

    // Add to deleted only if not in new list originally
    if !puow.isPaymentInList(payment, puow.deletedPayments) {
        puow.deletedPayments = append(puow.deletedPayments, payment)
        puow.logger.Debug("Registered deleted payment",
            zap.String("payment_id", payment.ID))
    }
}

// Commit all changes
func (puow *PaymentUnitOfWork) Commit(ctx context.Context) error {
    puow.mu.Lock()
    defer puow.mu.Unlock()

    if puow.isCommitted {
        return fmt.Errorf("unit of work already committed")
    }

    if puow.isRolledBack {
        return fmt.Errorf("unit of work already rolled back")
    }

    puow.logger.Info("Committing unit of work",
        zap.Int("new_payments", len(puow.newPayments)),
        zap.Int("modified_payments", len(puow.modifiedPayments)),
        zap.Int("deleted_payments", len(puow.deletedPayments)),
        zap.Int("new_accounts", len(puow.newAccounts)),
        zap.Int("modified_accounts", len(puow.modifiedAccounts)),
        zap.Int("new_transactions", len(puow.newTransactions)))

    // Set repositories to use our transaction
    puow.paymentRepo.SetTransaction(puow.dbTx)
    puow.accountRepo.SetTransaction(puow.dbTx)
    puow.transactionRepo.SetTransaction(puow.dbTx)

    // Commit in order: creates first, then updates, then deletes
    if err := puow.commitNewEntities(ctx); err != nil {
        puow.logger.Error("Failed to commit new entities", zap.Error(err))
        return puow.rollback(fmt.Errorf("commit new entities failed: %w", err))
    }

    if err := puow.commitModifiedEntities(ctx); err != nil {
        puow.logger.Error("Failed to commit modified entities", zap.Error(err))
        return puow.rollback(fmt.Errorf("commit modified entities failed: %w", err))
    }

    if err := puow.commitDeletedEntities(ctx); err != nil {
        puow.logger.Error("Failed to commit deleted entities", zap.Error(err))
        return puow.rollback(fmt.Errorf("commit deleted entities failed: %w", err))
    }

    // Commit database transaction
    if err := puow.dbTx.Commit(); err != nil {
        puow.logger.Error("Failed to commit database transaction", zap.Error(err))
        return fmt.Errorf("database commit failed: %w", err)
    }

    puow.isCommitted = true
    puow.logger.Info("Unit of work committed successfully")

    return nil
}

func (puow *PaymentUnitOfWork) commitNewEntities(ctx context.Context) error {
    // Create new accounts first (payments might reference them)
    for _, account := range puow.newAccounts {
        if err := puow.accountRepo.Create(ctx, account); err != nil {
            return fmt.Errorf("failed to create account %s: %w", account.ID, err)
        }
    }

    // Create new payments
    for _, payment := range puow.newPayments {
        if err := puow.paymentRepo.Create(ctx, payment); err != nil {
            return fmt.Errorf("failed to create payment %s: %w", payment.ID, err)
        }
    }

    // Create new transactions
    for _, transaction := range puow.newTransactions {
        if err := puow.transactionRepo.Create(ctx, transaction); err != nil {
            return fmt.Errorf("failed to create transaction %s: %w", transaction.ID, err)
        }
    }

    return nil
}

func (puow *PaymentUnitOfWork) commitModifiedEntities(ctx context.Context) error {
    // Update modified accounts
    for _, account := range puow.modifiedAccounts {
        if err := puow.accountRepo.Update(ctx, account); err != nil {
            return fmt.Errorf("failed to update account %s: %w", account.ID, err)
        }
    }

    // Update modified payments
    for _, payment := range puow.modifiedPayments {
        if err := puow.paymentRepo.Update(ctx, payment); err != nil {
            return fmt.Errorf("failed to update payment %s: %w", payment.ID, err)
        }
    }

    // Update modified transactions
    for _, transaction := range puow.modifiedTransactions {
        if err := puow.transactionRepo.Update(ctx, transaction); err != nil {
            return fmt.Errorf("failed to update transaction %s: %w", transaction.ID, err)
        }
    }

    return nil
}

func (puow *PaymentUnitOfWork) commitDeletedEntities(ctx context.Context) error {
    // Delete transactions first (might reference payments)
    for _, transaction := range puow.deletedTransactions {
        if err := puow.transactionRepo.Delete(ctx, transaction.ID); err != nil {
            return fmt.Errorf("failed to delete transaction %s: %w", transaction.ID, err)
        }
    }

    // Delete payments
    for _, payment := range puow.deletedPayments {
        if err := puow.paymentRepo.Delete(ctx, payment.ID); err != nil {
            return fmt.Errorf("failed to delete payment %s: %w", payment.ID, err)
        }
    }

    // Delete accounts last
    for _, account := range puow.deletedAccounts {
        if err := puow.accountRepo.Delete(ctx, account.ID); err != nil {
            return fmt.Errorf("failed to delete account %s: %w", account.ID, err)
        }
    }

    return nil
}

// Rollback all changes
func (puow *PaymentUnitOfWork) Rollback() error {
    puow.mu.Lock()
    defer puow.mu.Unlock()

    return puow.rollback(fmt.Errorf("explicit rollback"))
}

func (puow *PaymentUnitOfWork) rollback(cause error) error {
    if puow.isRolledBack {
        return fmt.Errorf("unit of work already rolled back")
    }

    if puow.isCommitted {
        return fmt.Errorf("cannot rollback committed unit of work")
    }

    puow.logger.Warn("Rolling back unit of work", zap.Error(cause))

    if err := puow.dbTx.Rollback(); err != nil {
        puow.logger.Error("Failed to rollback database transaction", zap.Error(err))
        return fmt.Errorf("database rollback failed: %w", err)
    }

    puow.isRolledBack = true
    puow.logger.Info("Unit of work rolled back")

    return cause
}

// Helper methods
func (puow *PaymentUnitOfWork) isPaymentInList(payment *PaymentEntity, list []*PaymentEntity) bool {
    for _, p := range list {
        if p.ID == payment.ID {
            return true
        }
    }
    return false
}

func (puow *PaymentUnitOfWork) isAccountInList(account *AccountEntity, list []*AccountEntity) bool {
    for _, a := range list {
        if a.ID == account.ID {
            return true
        }
    }
    return false
}

func (puow *PaymentUnitOfWork) removePaymentFromList(payment *PaymentEntity, list *[]*PaymentEntity) {
    for i, p := range *list {
        if p.ID == payment.ID {
            *list = append((*list)[:i], (*list)[i+1:]...)
            break
        }
    }
}

// Get statistics
func (puow *PaymentUnitOfWork) GetStatistics() UnitOfWorkStatistics {
    puow.mu.RLock()
    defer puow.mu.RUnlock()

    return UnitOfWorkStatistics{
        NewEntities:      len(puow.newPayments) + len(puow.newAccounts) + len(puow.newTransactions),
        ModifiedEntities: len(puow.modifiedPayments) + len(puow.modifiedAccounts) + len(puow.modifiedTransactions),
        DeletedEntities:  len(puow.deletedPayments) + len(puow.deletedAccounts) + len(puow.deletedTransactions),
        IsCommitted:      puow.isCommitted,
        IsRolledBack:     puow.isRolledBack,
    }
}

// Entity definitions
type PaymentEntity struct {
    ID            string
    Amount        decimal.Decimal
    Currency      string
    Status        string
    CustomerID    string
    MerchantID    string
    PaymentMethod string
    CreatedAt     time.Time
    UpdatedAt     time.Time
    Version       int64 // For optimistic locking
}

type AccountEntity struct {
    ID         string
    CustomerID string
    Balance    decimal.Decimal
    Currency   string
    Status     string
    CreatedAt  time.Time
    UpdatedAt  time.Time
    Version    int64
}

type TransactionEntity struct {
    ID          string
    Type        string
    Amount      decimal.Decimal
    Currency    string
    Status      string
    PaymentID   string
    AccountID   string
    Description string
    CreatedAt   time.Time
    Version     int64
}

type UnitOfWorkStatistics struct {
    NewEntities      int
    ModifiedEntities int
    DeletedEntities  int
    IsCommitted      bool
    IsRolledBack     bool
}

// Repository interfaces
type PaymentRepository interface {
    Create(ctx context.Context, payment *PaymentEntity) error
    Update(ctx context.Context, payment *PaymentEntity) error
    Delete(ctx context.Context, id string) error
    FindByID(ctx context.Context, id string) (*PaymentEntity, error)
    SetTransaction(tx DatabaseTransaction)
}

type AccountRepository interface {
    Create(ctx context.Context, account *AccountEntity) error
    Update(ctx context.Context, account *AccountEntity) error
    Delete(ctx context.Context, id string) error
    FindByID(ctx context.Context, id string) (*AccountEntity, error)
    SetTransaction(tx DatabaseTransaction)
}

type TransactionRepository interface {
    Create(ctx context.Context, transaction *TransactionEntity) error
    Update(ctx context.Context, transaction *TransactionEntity) error
    Delete(ctx context.Context, id string) error
    FindByID(ctx context.Context, id string) (*TransactionEntity, error)
    SetTransaction(tx DatabaseTransaction)
}

type DatabaseTransaction interface {
    Commit() error
    Rollback() error
    IsActive() bool
}

type Database interface {
    BeginTransaction() (DatabaseTransaction, error)
}
```

### 2. Trading Settlement Unit of Work

```go
// Trading settlement involves multiple trades, accounts, and positions
type SettlementUnitOfWork struct {
    // Trades to settle
    newTrades        []*TradeEntity
    modifiedTrades   []*TradeEntity

    // Positions to update
    modifiedPositions []*PositionEntity

    // Account balance changes
    balanceChanges   []*BalanceChangeEntity

    // Settlement records
    newSettlements   []*SettlementEntity

    // Repositories
    tradeRepo      TradeRepository
    positionRepo   PositionRepository
    accountRepo    AccountRepository
    settlementRepo SettlementRepository

    // Services
    clearingService ClearingService
    riskService     RiskService

    dbTx   DatabaseTransaction
    logger *zap.Logger

    isCommitted  bool
    isRolledBack bool
    mu           sync.RWMutex
}

func NewSettlementUnitOfWork(
    repos SettlementRepositories,
    services SettlementServices,
    db Database,
    logger *zap.Logger,
) (*SettlementUnitOfWork, error) {
    dbTx, err := db.BeginTransaction()
    if err != nil {
        return nil, fmt.Errorf("failed to begin settlement transaction: %w", err)
    }

    return &SettlementUnitOfWork{
        newTrades:         make([]*TradeEntity, 0),
        modifiedTrades:    make([]*TradeEntity, 0),
        modifiedPositions: make([]*PositionEntity, 0),
        balanceChanges:    make([]*BalanceChangeEntity, 0),
        newSettlements:    make([]*SettlementEntity, 0),
        tradeRepo:         repos.TradeRepo,
        positionRepo:      repos.PositionRepo,
        accountRepo:       repos.AccountRepo,
        settlementRepo:    repos.SettlementRepo,
        clearingService:   services.ClearingService,
        riskService:       services.RiskService,
        dbTx:             dbTx,
        logger:           logger,
    }, nil
}

// Process trade settlement
func (suow *SettlementUnitOfWork) ProcessTradeSettlement(ctx context.Context, trade *TradeEntity) error {
    suow.mu.Lock()
    defer suow.mu.Unlock()

    if suow.isCommitted || suow.isRolledBack {
        return fmt.Errorf("cannot process trade on completed unit of work")
    }

    suow.logger.Info("Processing trade settlement",
        zap.String("trade_id", trade.ID),
        zap.String("symbol", trade.Symbol),
        zap.String("quantity", trade.Quantity.String()))

    // 1. Validate trade for settlement
    if err := suow.validateTradeForSettlement(trade); err != nil {
        return fmt.Errorf("trade validation failed: %w", err)
    }

    // 2. Calculate settlement details
    settlement, err := suow.calculateSettlement(trade)
    if err != nil {
        return fmt.Errorf("settlement calculation failed: %w", err)
    }

    // 3. Update positions
    if err := suow.updatePositions(trade); err != nil {
        return fmt.Errorf("position update failed: %w", err)
    }

    // 4. Calculate balance changes
    if err := suow.calculateBalanceChanges(trade, settlement); err != nil {
        return fmt.Errorf("balance calculation failed: %w", err)
    }

    // 5. Register entities for persistence
    trade.Status = "SETTLED"
    trade.SettlementDate = time.Now()
    suow.registerModifiedTrade(trade)
    suow.registerNewSettlement(settlement)

    return nil
}

func (suow *SettlementUnitOfWork) validateTradeForSettlement(trade *TradeEntity) error {
    if trade.Status != "EXECUTED" {
        return fmt.Errorf("trade %s not in EXECUTED status: %s", trade.ID, trade.Status)
    }

    if trade.Quantity.LessThanOrEqual(decimal.Zero) {
        return fmt.Errorf("invalid trade quantity: %s", trade.Quantity.String())
    }

    // Risk checks
    if err := suow.riskService.ValidateSettlement(trade); err != nil {
        return fmt.Errorf("risk validation failed: %w", err)
    }

    return nil
}

func (suow *SettlementUnitOfWork) calculateSettlement(trade *TradeEntity) (*SettlementEntity, error) {
    settlement := &SettlementEntity{
        ID:           generateSettlementID(),
        TradeID:      trade.ID,
        Symbol:       trade.Symbol,
        Quantity:     trade.Quantity,
        Price:        trade.Price,
        GrossAmount:  trade.Quantity.Mul(trade.Price),
        SettlementDate: time.Now().AddDate(0, 0, 2), // T+2 settlement
        Currency:     trade.Currency,
        Status:       "PENDING",
        CreatedAt:    time.Now(),
    }

    // Calculate fees and net amount
    fees := suow.clearingService.CalculateFees(trade)
    settlement.Fees = fees.Total
    settlement.NetAmount = settlement.GrossAmount.Sub(fees.Total)

    // Add clearing details
    clearingDetails, err := suow.clearingService.GetClearingDetails(trade)
    if err != nil {
        return nil, fmt.Errorf("failed to get clearing details: %w", err)
    }
    settlement.ClearingHouse = clearingDetails.ClearingHouse
    settlement.ClearingNumber = clearingDetails.ClearingNumber

    return settlement, nil
}

func (suow *SettlementUnitOfWork) updatePositions(trade *TradeEntity) error {
    // Get or create position for the symbol
    position, err := suow.positionRepo.FindBySymbol(context.Background(), trade.BuyerID, trade.Symbol)
    if err != nil {
        // Create new position
        position = &PositionEntity{
            ID:         generatePositionID(),
            AccountID:  trade.BuyerID,
            Symbol:     trade.Symbol,
            Quantity:   decimal.Zero,
            AvgCost:    decimal.Zero,
            MarketValue: decimal.Zero,
            UpdatedAt:  time.Now(),
        }
    }

    // Update position based on trade side
    if trade.Side == "BUY" {
        // Calculate new average cost
        totalCost := position.Quantity.Mul(position.AvgCost).Add(trade.Quantity.Mul(trade.Price))
        newQuantity := position.Quantity.Add(trade.Quantity)
        position.AvgCost = totalCost.Div(newQuantity)
        position.Quantity = newQuantity
    } else { // SELL
        position.Quantity = position.Quantity.Sub(trade.Quantity)
        // Average cost remains the same for sells
    }

    position.UpdatedAt = time.Now()
    position.Version++

    suow.registerModifiedPosition(position)

    return nil
}

func (suow *SettlementUnitOfWork) calculateBalanceChanges(trade *TradeEntity, settlement *SettlementEntity) error {
    // Buyer balance change (cash out, securities in)
    buyerCashChange := &BalanceChangeEntity{
        ID:          generateBalanceChangeID(),
        AccountID:   trade.BuyerID,
        Currency:    trade.Currency,
        Amount:      settlement.NetAmount.Neg(), // Cash out
        Type:        "TRADE_SETTLEMENT",
        Reference:   trade.ID,
        Description: fmt.Sprintf("Purchase of %s %s", trade.Quantity.String(), trade.Symbol),
        CreatedAt:   time.Now(),
    }

    // Seller balance change (cash in, securities out)
    sellerCashChange := &BalanceChangeEntity{
        ID:          generateBalanceChangeID(),
        AccountID:   trade.SellerID,
        Currency:    trade.Currency,
        Amount:      settlement.NetAmount, // Cash in
        Type:        "TRADE_SETTLEMENT",
        Reference:   trade.ID,
        Description: fmt.Sprintf("Sale of %s %s", trade.Quantity.String(), trade.Symbol),
        CreatedAt:   time.Now(),
    }

    suow.balanceChanges = append(suow.balanceChanges, buyerCashChange, sellerCashChange)

    return nil
}

// Registration methods
func (suow *SettlementUnitOfWork) registerModifiedTrade(trade *TradeEntity) {
    suow.modifiedTrades = append(suow.modifiedTrades, trade)
}

func (suow *SettlementUnitOfWork) registerModifiedPosition(position *PositionEntity) {
    // Check if already registered
    for _, p := range suow.modifiedPositions {
        if p.ID == position.ID {
            return
        }
    }
    suow.modifiedPositions = append(suow.modifiedPositions, position)
}

func (suow *SettlementUnitOfWork) registerNewSettlement(settlement *SettlementEntity) {
    suow.newSettlements = append(suow.newSettlements, settlement)
}

// Commit settlement
func (suow *SettlementUnitOfWork) Commit(ctx context.Context) error {
    suow.mu.Lock()
    defer suow.mu.Unlock()

    if suow.isCommitted {
        return fmt.Errorf("settlement unit of work already committed")
    }

    if suow.isRolledBack {
        return fmt.Errorf("settlement unit of work already rolled back")
    }

    suow.logger.Info("Committing settlement unit of work",
        zap.Int("trades", len(suow.modifiedTrades)),
        zap.Int("positions", len(suow.modifiedPositions)),
        zap.Int("settlements", len(suow.newSettlements)),
        zap.Int("balance_changes", len(suow.balanceChanges)))

    // Set transaction context for all repositories
    suow.setTransactionContext()

    // Commit in order
    if err := suow.commitSettlements(ctx); err != nil {
        return suow.rollback(fmt.Errorf("failed to commit settlements: %w", err))
    }

    if err := suow.commitTrades(ctx); err != nil {
        return suow.rollback(fmt.Errorf("failed to commit trades: %w", err))
    }

    if err := suow.commitPositions(ctx); err != nil {
        return suow.rollback(fmt.Errorf("failed to commit positions: %w", err))
    }

    if err := suow.commitBalanceChanges(ctx); err != nil {
        return suow.rollback(fmt.Errorf("failed to commit balance changes: %w", err))
    }

    // Final database commit
    if err := suow.dbTx.Commit(); err != nil {
        return fmt.Errorf("database commit failed: %w", err)
    }

    suow.isCommitted = true
    suow.logger.Info("Settlement unit of work committed successfully")

    return nil
}

func (suow *SettlementUnitOfWork) setTransactionContext() {
    suow.tradeRepo.SetTransaction(suow.dbTx)
    suow.positionRepo.SetTransaction(suow.dbTx)
    suow.accountRepo.SetTransaction(suow.dbTx)
    suow.settlementRepo.SetTransaction(suow.dbTx)
}

func (suow *SettlementUnitOfWork) commitSettlements(ctx context.Context) error {
    for _, settlement := range suow.newSettlements {
        if err := suow.settlementRepo.Create(ctx, settlement); err != nil {
            return fmt.Errorf("failed to create settlement %s: %w", settlement.ID, err)
        }
    }
    return nil
}

func (suow *SettlementUnitOfWork) commitTrades(ctx context.Context) error {
    for _, trade := range suow.modifiedTrades {
        if err := suow.tradeRepo.Update(ctx, trade); err != nil {
            return fmt.Errorf("failed to update trade %s: %w", trade.ID, err)
        }
    }
    return nil
}

func (suow *SettlementUnitOfWork) commitPositions(ctx context.Context) error {
    for _, position := range suow.modifiedPositions {
        if err := suow.positionRepo.Update(ctx, position); err != nil {
            return fmt.Errorf("failed to update position %s: %w", position.ID, err)
        }
    }
    return nil
}

func (suow *SettlementUnitOfWork) commitBalanceChanges(ctx context.Context) error {
    for _, change := range suow.balanceChanges {
        if err := suow.accountRepo.ApplyBalanceChange(ctx, change); err != nil {
            return fmt.Errorf("failed to apply balance change %s: %w", change.ID, err)
        }
    }
    return nil
}

func (suow *SettlementUnitOfWork) rollback(cause error) error {
    if err := suow.dbTx.Rollback(); err != nil {
        suow.logger.Error("Failed to rollback settlement transaction", zap.Error(err))
        return fmt.Errorf("settlement rollback failed: %w", err)
    }

    suow.isRolledBack = true
    suow.logger.Warn("Settlement unit of work rolled back", zap.Error(cause))

    return cause
}

// Supporting types for settlement
type TradeEntity struct {
    ID             string
    Symbol         string
    Quantity       decimal.Decimal
    Price          decimal.Decimal
    Side           string // BUY/SELL
    BuyerID        string
    SellerID       string
    Currency       string
    Status         string
    ExecutedAt     time.Time
    SettlementDate time.Time
    Version        int64
}

type PositionEntity struct {
    ID          string
    AccountID   string
    Symbol      string
    Quantity    decimal.Decimal
    AvgCost     decimal.Decimal
    MarketValue decimal.Decimal
    UpdatedAt   time.Time
    Version     int64
}

type SettlementEntity struct {
    ID             string
    TradeID        string
    Symbol         string
    Quantity       decimal.Decimal
    Price          decimal.Decimal
    GrossAmount    decimal.Decimal
    Fees           decimal.Decimal
    NetAmount      decimal.Decimal
    Currency       string
    SettlementDate time.Time
    ClearingHouse  string
    ClearingNumber string
    Status         string
    CreatedAt      time.Time
}

type BalanceChangeEntity struct {
    ID          string
    AccountID   string
    Currency    string
    Amount      decimal.Decimal
    Type        string
    Reference   string
    Description string
    CreatedAt   time.Time
}

// Repository and service interfaces for settlement
type SettlementRepositories struct {
    TradeRepo      TradeRepository
    PositionRepo   PositionRepository
    AccountRepo    AccountRepository
    SettlementRepo SettlementRepository
}

type SettlementServices struct {
    ClearingService ClearingService
    RiskService     RiskService
}

type TradeRepository interface {
    Update(ctx context.Context, trade *TradeEntity) error
    SetTransaction(tx DatabaseTransaction)
}

type PositionRepository interface {
    FindBySymbol(ctx context.Context, accountID, symbol string) (*PositionEntity, error)
    Update(ctx context.Context, position *PositionEntity) error
    SetTransaction(tx DatabaseTransaction)
}

type SettlementRepository interface {
    Create(ctx context.Context, settlement *SettlementEntity) error
    SetTransaction(tx DatabaseTransaction)
}

type ClearingService interface {
    CalculateFees(trade *TradeEntity) *TradeFees
    GetClearingDetails(trade *TradeEntity) (*ClearingDetails, error)
}

type RiskService interface {
    ValidateSettlement(trade *TradeEntity) error
}

type TradeFees struct {
    Commission    decimal.Decimal
    ClearingFee   decimal.Decimal
    RegulatoryFee decimal.Decimal
    Total         decimal.Decimal
}

type ClearingDetails struct {
    ClearingHouse  string
    ClearingNumber string
}

// Helper functions
func generateSettlementID() string {
    return fmt.Sprintf("SETTLE_%d_%s", time.Now().Unix(), randomString(8))
}

func generatePositionID() string {
    return fmt.Sprintf("POS_%d_%s", time.Now().Unix(), randomString(8))
}

func generateBalanceChangeID() string {
    return fmt.Sprintf("BAL_%d_%s", time.Now().Unix(), randomString(8))
}

func randomString(length int) string {
    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    result := make([]byte, length)
    for i := range result {
        result[i] = chars[rand.Intn(len(chars))]
    }
    return string(result)
}
```

## Go Implementation

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
    "go.uber.org/zap"
    "github.com/shopspring/decimal"
)

// Simple Unit of Work implementation for demonstration
// Shows basic change tracking and transaction coordination

// Entity interface
type Entity interface {
    GetID() string
    GetVersion() int64
    SetVersion(version int64)
    Clone() Entity
}

// Unit of Work interface
type UnitOfWork interface {
    RegisterNew(entity Entity)
    RegisterModified(entity Entity)
    RegisterDeleted(entity Entity)
    Commit(ctx context.Context) error
    Rollback() error
    GetStatistics() UnitOfWorkStatistics
}

// Basic Unit of Work implementation
type BasicUnitOfWork struct {
    newEntities      map[string]Entity
    modifiedEntities map[string]Entity
    deletedEntities  map[string]Entity

    repositories map[string]Repository
    transaction  Transaction
    logger       *zap.Logger

    isCommitted  bool
    isRolledBack bool
    mu           sync.RWMutex
}

func NewBasicUnitOfWork(repos map[string]Repository, tx Transaction, logger *zap.Logger) *BasicUnitOfWork {
    return &BasicUnitOfWork{
        newEntities:      make(map[string]Entity),
        modifiedEntities: make(map[string]Entity),
        deletedEntities:  make(map[string]Entity),
        repositories:     repos,
        transaction:      tx,
        logger:          logger,
    }
}

func (uow *BasicUnitOfWork) RegisterNew(entity Entity) {
    uow.mu.Lock()
    defer uow.mu.Unlock()

    if uow.isCommitted || uow.isRolledBack {
        uow.logger.Warn("Attempting to register entity on completed unit of work")
        return
    }

    entityID := entity.GetID()

    // Remove from other collections if present
    delete(uow.modifiedEntities, entityID)
    delete(uow.deletedEntities, entityID)

    uow.newEntities[entityID] = entity
    uow.logger.Debug("Registered new entity", zap.String("entity_id", entityID))
}

func (uow *BasicUnitOfWork) RegisterModified(entity Entity) {
    uow.mu.Lock()
    defer uow.mu.Unlock()

    if uow.isCommitted || uow.isRolledBack {
        uow.logger.Warn("Attempting to register entity on completed unit of work")
        return
    }

    entityID := entity.GetID()

    // Only register as modified if not already new
    if _, isNew := uow.newEntities[entityID]; !isNew {
        uow.modifiedEntities[entityID] = entity
        uow.logger.Debug("Registered modified entity", zap.String("entity_id", entityID))
    }
}

func (uow *BasicUnitOfWork) RegisterDeleted(entity Entity) {
    uow.mu.Lock()
    defer uow.mu.Unlock()

    if uow.isCommitted || uow.isRolledBack {
        uow.logger.Warn("Attempting to register entity on completed unit of work")
        return
    }

    entityID := entity.GetID()

    // If entity was new, just remove it
    if _, isNew := uow.newEntities[entityID]; isNew {
        delete(uow.newEntities, entityID)
    } else {
        // Remove from modified and add to deleted
        delete(uow.modifiedEntities, entityID)
        uow.deletedEntities[entityID] = entity
    }

    uow.logger.Debug("Registered deleted entity", zap.String("entity_id", entityID))
}

func (uow *BasicUnitOfWork) Commit(ctx context.Context) error {
    uow.mu.Lock()
    defer uow.mu.Unlock()

    if uow.isCommitted {
        return fmt.Errorf("unit of work already committed")
    }

    if uow.isRolledBack {
        return fmt.Errorf("unit of work already rolled back")
    }

    uow.logger.Info("Committing unit of work",
        zap.Int("new_entities", len(uow.newEntities)),
        zap.Int("modified_entities", len(uow.modifiedEntities)),
        zap.Int("deleted_entities", len(uow.deletedEntities)))

    // Set transaction context for all repositories
    for _, repo := range uow.repositories {
        repo.SetTransaction(uow.transaction)
    }

    // Commit in order: new, modified, deleted
    if err := uow.commitNewEntities(ctx); err != nil {
        return uow.rollback(fmt.Errorf("failed to commit new entities: %w", err))
    }

    if err := uow.commitModifiedEntities(ctx); err != nil {
        return uow.rollback(fmt.Errorf("failed to commit modified entities: %w", err))
    }

    if err := uow.commitDeletedEntities(ctx); err != nil {
        return uow.rollback(fmt.Errorf("failed to commit deleted entities: %w", err))
    }

    // Commit the transaction
    if err := uow.transaction.Commit(); err != nil {
        return fmt.Errorf("transaction commit failed: %w", err)
    }

    uow.isCommitted = true
    uow.logger.Info("Unit of work committed successfully")

    return nil
}

func (uow *BasicUnitOfWork) commitNewEntities(ctx context.Context) error {
    for _, entity := range uow.newEntities {
        repo := uow.getRepositoryForEntity(entity)
        if repo == nil {
            return fmt.Errorf("no repository found for entity type: %T", entity)
        }

        if err := repo.Create(ctx, entity); err != nil {
            return fmt.Errorf("failed to create entity %s: %w", entity.GetID(), err)
        }
    }
    return nil
}

func (uow *BasicUnitOfWork) commitModifiedEntities(ctx context.Context) error {
    for _, entity := range uow.modifiedEntities {
        repo := uow.getRepositoryForEntity(entity)
        if repo == nil {
            return fmt.Errorf("no repository found for entity type: %T", entity)
        }

        if err := repo.Update(ctx, entity); err != nil {
            return fmt.Errorf("failed to update entity %s: %w", entity.GetID(), err)
        }
    }
    return nil
}

func (uow *BasicUnitOfWork) commitDeletedEntities(ctx context.Context) error {
    for _, entity := range uow.deletedEntities {
        repo := uow.getRepositoryForEntity(entity)
        if repo == nil {
            return fmt.Errorf("no repository found for entity type: %T", entity)
        }

        if err := repo.Delete(ctx, entity.GetID()); err != nil {
            return fmt.Errorf("failed to delete entity %s: %w", entity.GetID(), err)
        }
    }
    return nil
}

func (uow *BasicUnitOfWork) getRepositoryForEntity(entity Entity) Repository {
    switch entity.(type) {
    case *User:
        return uow.repositories["user"]
    case *Order:
        return uow.repositories["order"]
    case *Product:
        return uow.repositories["product"]
    default:
        return nil
    }
}

func (uow *BasicUnitOfWork) Rollback() error {
    uow.mu.Lock()
    defer uow.mu.Unlock()

    return uow.rollback(fmt.Errorf("explicit rollback"))
}

func (uow *BasicUnitOfWork) rollback(cause error) error {
    if uow.isRolledBack {
        return fmt.Errorf("unit of work already rolled back")
    }

    if uow.isCommitted {
        return fmt.Errorf("cannot rollback committed unit of work")
    }

    uow.logger.Warn("Rolling back unit of work", zap.Error(cause))

    if err := uow.transaction.Rollback(); err != nil {
        return fmt.Errorf("transaction rollback failed: %w", err)
    }

    uow.isRolledBack = true
    uow.logger.Info("Unit of work rolled back")

    return cause
}

func (uow *BasicUnitOfWork) GetStatistics() UnitOfWorkStatistics {
    uow.mu.RLock()
    defer uow.mu.RUnlock()

    return UnitOfWorkStatistics{
        NewEntities:      len(uow.newEntities),
        ModifiedEntities: len(uow.modifiedEntities),
        DeletedEntities:  len(uow.deletedEntities),
        IsCommitted:      uow.isCommitted,
        IsRolledBack:     uow.isRolledBack,
    }
}

// Sample entities
type User struct {
    ID       string
    Name     string
    Email    string
    Version  int64
    CreatedAt time.Time
    UpdatedAt time.Time
}

func (u *User) GetID() string {
    return u.ID
}

func (u *User) GetVersion() int64 {
    return u.Version
}

func (u *User) SetVersion(version int64) {
    u.Version = version
}

func (u *User) Clone() Entity {
    clone := *u
    return &clone
}

type Order struct {
    ID        string
    UserID    string
    Total     decimal.Decimal
    Status    string
    Version   int64
    CreatedAt time.Time
    UpdatedAt time.Time
}

func (o *Order) GetID() string {
    return o.ID
}

func (o *Order) GetVersion() int64 {
    return o.Version
}

func (o *Order) SetVersion(version int64) {
    o.Version = version
}

func (o *Order) Clone() Entity {
    clone := *o
    return &clone
}

type Product struct {
    ID          string
    Name        string
    Price       decimal.Decimal
    Stock       int
    Version     int64
    CreatedAt   time.Time
    UpdatedAt   time.Time
}

func (p *Product) GetID() string {
    return p.ID
}

func (p *Product) GetVersion() int64 {
    return p.Version
}

func (p *Product) SetVersion(version int64) {
    p.Version = version
}

func (p *Product) Clone() Entity {
    clone := *p
    return &clone
}

// Repository interface
type Repository interface {
    Create(ctx context.Context, entity Entity) error
    Update(ctx context.Context, entity Entity) error
    Delete(ctx context.Context, id string) error
    FindByID(ctx context.Context, id string) (Entity, error)
    SetTransaction(tx Transaction)
}

// Mock repository for demonstration
type MockRepository struct {
    entityType string
    data       map[string]Entity
    transaction Transaction
    logger     *zap.Logger
    mu         sync.RWMutex
}

func NewMockRepository(entityType string, logger *zap.Logger) *MockRepository {
    return &MockRepository{
        entityType: entityType,
        data:       make(map[string]Entity),
        logger:     logger,
    }
}

func (mr *MockRepository) Create(ctx context.Context, entity Entity) error {
    mr.mu.Lock()
    defer mr.mu.Unlock()

    if mr.transaction == nil || !mr.transaction.IsActive() {
        return fmt.Errorf("no active transaction")
    }

    entityID := entity.GetID()
    if _, exists := mr.data[entityID]; exists {
        return fmt.Errorf("entity %s already exists", entityID)
    }

    // Clone entity to avoid external modifications
    mr.data[entityID] = entity.Clone()
    mr.logger.Debug("Created entity",
        zap.String("type", mr.entityType),
        zap.String("id", entityID))

    return nil
}

func (mr *MockRepository) Update(ctx context.Context, entity Entity) error {
    mr.mu.Lock()
    defer mr.mu.Unlock()

    if mr.transaction == nil || !mr.transaction.IsActive() {
        return fmt.Errorf("no active transaction")
    }

    entityID := entity.GetID()
    existing, exists := mr.data[entityID]
    if !exists {
        return fmt.Errorf("entity %s not found", entityID)
    }

    // Optimistic locking check
    if existing.GetVersion() != entity.GetVersion() {
        return fmt.Errorf("optimistic locking failed for entity %s: expected version %d, got %d",
            entityID, existing.GetVersion(), entity.GetVersion())
    }

    // Increment version
    entity.SetVersion(entity.GetVersion() + 1)
    mr.data[entityID] = entity.Clone()

    mr.logger.Debug("Updated entity",
        zap.String("type", mr.entityType),
        zap.String("id", entityID),
        zap.Int64("new_version", entity.GetVersion()))

    return nil
}

func (mr *MockRepository) Delete(ctx context.Context, id string) error {
    mr.mu.Lock()
    defer mr.mu.Unlock()

    if mr.transaction == nil || !mr.transaction.IsActive() {
        return fmt.Errorf("no active transaction")
    }

    if _, exists := mr.data[id]; !exists {
        return fmt.Errorf("entity %s not found", id)
    }

    delete(mr.data, id)
    mr.logger.Debug("Deleted entity",
        zap.String("type", mr.entityType),
        zap.String("id", id))

    return nil
}

func (mr *MockRepository) FindByID(ctx context.Context, id string) (Entity, error) {
    mr.mu.RLock()
    defer mr.mu.RUnlock()

    entity, exists := mr.data[id]
    if !exists {
        return nil, fmt.Errorf("entity %s not found", id)
    }

    return entity.Clone(), nil
}

func (mr *MockRepository) SetTransaction(tx Transaction) {
    mr.transaction = tx
}

// Transaction interface
type Transaction interface {
    Commit() error
    Rollback() error
    IsActive() bool
}

// Mock transaction
type MockTransaction struct {
    isActive     bool
    isCommitted  bool
    isRolledBack bool
    logger       *zap.Logger
    mu           sync.RWMutex
}

func NewMockTransaction(logger *zap.Logger) *MockTransaction {
    return &MockTransaction{
        isActive: true,
        logger:   logger,
    }
}

func (mt *MockTransaction) Commit() error {
    mt.mu.Lock()
    defer mt.mu.Unlock()

    if !mt.isActive {
        return fmt.Errorf("transaction not active")
    }

    if mt.isCommitted {
        return fmt.Errorf("transaction already committed")
    }

    if mt.isRolledBack {
        return fmt.Errorf("transaction already rolled back")
    }

    mt.isCommitted = true
    mt.isActive = false
    mt.logger.Info("Transaction committed")

    return nil
}

func (mt *MockTransaction) Rollback() error {
    mt.mu.Lock()
    defer mt.mu.Unlock()

    if !mt.isActive {
        return fmt.Errorf("transaction not active")
    }

    if mt.isCommitted {
        return fmt.Errorf("cannot rollback committed transaction")
    }

    if mt.isRolledBack {
        return fmt.Errorf("transaction already rolled back")
    }

    mt.isRolledBack = true
    mt.isActive = false
    mt.logger.Info("Transaction rolled back")

    return nil
}

func (mt *MockTransaction) IsActive() bool {
    mt.mu.RLock()
    defer mt.mu.RUnlock()
    return mt.isActive
}

// Example usage
func main() {
    fmt.Println("=== Unit of Work Pattern Demo ===\n")

    // Create logger
    logger, _ := zap.NewDevelopment()
    defer logger.Sync()

    // Create repositories
    repositories := map[string]Repository{
        "user":    NewMockRepository("User", logger),
        "order":   NewMockRepository("Order", logger),
        "product": NewMockRepository("Product", logger),
    }

    // Create transaction
    transaction := NewMockTransaction(logger)

    // Create unit of work
    uow := NewBasicUnitOfWork(repositories, transaction, logger)

    // Example 1: Create new entities
    fmt.Println("=== Creating Entities ===")

    user := &User{
        ID:        "user_1",
        Name:      "John Doe",
        Email:     "john@example.com",
        Version:   0,
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }

    product := &Product{
        ID:        "product_1",
        Name:      "Laptop",
        Price:     decimal.NewFromInt(999),
        Stock:     10,
        Version:   0,
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }

    order := &Order{
        ID:        "order_1",
        UserID:    user.ID,
        Total:     product.Price,
        Status:    "PENDING",
        Version:   0,
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }

    // Register entities with unit of work
    uow.RegisterNew(user)
    uow.RegisterNew(product)
    uow.RegisterNew(order)

    fmt.Printf("Registered %d new entities\n", uow.GetStatistics().NewEntities)

    // Example 2: Modify entities
    fmt.Println("\n=== Modifying Entities ===")

    order.Status = "CONFIRMED"
    order.UpdatedAt = time.Now()
    uow.RegisterModified(order)

    product.Stock = 9 // Reduce stock
    product.UpdatedAt = time.Now()
    uow.RegisterModified(product)

    stats := uow.GetStatistics()
    fmt.Printf("Statistics: %d new, %d modified, %d deleted\n",
        stats.NewEntities, stats.ModifiedEntities, stats.DeletedEntities)

    // Example 3: Commit unit of work
    fmt.Println("\n=== Committing Unit of Work ===")

    if err := uow.Commit(context.Background()); err != nil {
        fmt.Printf("Commit failed: %v\n", err)
    } else {
        fmt.Println("Unit of work committed successfully!")
    }

    finalStats := uow.GetStatistics()
    fmt.Printf("Final state: committed=%t, rolled_back=%t\n",
        finalStats.IsCommitted, finalStats.IsRolledBack)

    // Example 4: Demonstrate rollback scenario
    fmt.Println("\n=== Rollback Scenario ===")

    // Create new transaction and unit of work
    transaction2 := NewMockTransaction(logger)
    uow2 := NewBasicUnitOfWork(repositories, transaction2, logger)

    // Create an entity that will cause an error (duplicate ID)
    duplicateUser := &User{
        ID:        "user_1", // Same ID as before
        Name:      "Jane Doe",
        Email:     "jane@example.com",
        Version:   0,
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }

    uow2.RegisterNew(duplicateUser)

    if err := uow2.Commit(context.Background()); err != nil {
        fmt.Printf("Expected error occurred: %v\n", err)
        rollbackStats := uow2.GetStatistics()
        fmt.Printf("Rollback state: committed=%t, rolled_back=%t\n",
            rollbackStats.IsCommitted, rollbackStats.IsRolledBack)
    }

    // Example 5: Demonstrate optimistic locking
    fmt.Println("\n=== Optimistic Locking Demo ===")

    transaction3 := NewMockTransaction(logger)
    uow3 := NewBasicUnitOfWork(repositories, transaction3, logger)

    // Try to modify user with wrong version
    userCopy := &User{
        ID:        "user_1",
        Name:      "John Smith", // Changed name
        Email:     "john@example.com",
        Version:   0, // Wrong version (should be 1 after previous update)
        UpdatedAt: time.Now(),
    }

    uow3.RegisterModified(userCopy)

    if err := uow3.Commit(context.Background()); err != nil {
        fmt.Printf("Optimistic locking error: %v\n", err)
    }

    fmt.Println("\n=== Unit of Work Pattern Demo Complete ===")
}
```

## Variants & Trade-offs

### Variants

1. **Identity Map Integration**

```go
type IdentityMapUnitOfWork struct {
    *BasicUnitOfWork
    identityMap map[string]Entity
}

func (imuow *IdentityMapUnitOfWork) GetEntity(id string) Entity {
    if entity, exists := imuow.identityMap[id]; exists {
        return entity
    }
    return nil
}

func (imuow *IdentityMapUnitOfWork) RegisterEntity(entity Entity) {
    imuow.identityMap[entity.GetID()] = entity
}
```

2. **Change Tracking Unit of Work**

```go
type ChangeTrackingUnitOfWork struct {
    *BasicUnitOfWork
    originalValues map[string]Entity
}

func (ctuow *ChangeTrackingUnitOfWork) StartTracking(entity Entity) {
    ctuow.originalValues[entity.GetID()] = entity.Clone()
}

func (ctuow *ChangeTrackingUnitOfWork) GetChanges(entity Entity) map[string]interface{} {
    original := ctuow.originalValues[entity.GetID()]
    return calculateDifferences(original, entity)
}
```

3. **Async Unit of Work**

```go
type AsyncUnitOfWork struct {
    *BasicUnitOfWork
    commitChannel chan commitRequest
    workers       int
}

type commitRequest struct {
    entities []Entity
    result   chan error
}

func (auow *AsyncUnitOfWork) CommitAsync(ctx context.Context) <-chan error {
    result := make(chan error, 1)

    go func() {
        result <- auow.BasicUnitOfWork.Commit(ctx)
    }()

    return result
}
```

### Trade-offs

**Pros:**

- **Transactional Consistency**: Ensures ACID properties across operations
- **Performance**: Batches database operations for efficiency
- **Change Tracking**: Automatically tracks what needs to be persisted
- **Rollback Support**: Easy to undo changes on failure
- **Decoupling**: Separates business logic from persistence concerns

**Cons:**

- **Memory Usage**: Holds entities in memory until commit
- **Complexity**: Adds complexity to simple operations
- **Concurrency**: Can create contention in multi-threaded scenarios
- **State Management**: Requires careful state tracking
- **Learning Curve**: More complex than direct repository usage

## Integration Tips

### 1. Repository Pattern Integration

```go
type UnitOfWorkRepository struct {
    baseRepo Repository
    uow      UnitOfWork
}

func (uowr *UnitOfWorkRepository) Save(entity Entity) error {
    if entity.GetVersion() == 0 {
        uowr.uow.RegisterNew(entity)
    } else {
        uowr.uow.RegisterModified(entity)
    }
    return nil
}

func (uowr *UnitOfWorkRepository) Delete(entity Entity) error {
    uowr.uow.RegisterDeleted(entity)
    return nil
}
```

### 2. Domain Event Integration

```go
type EventAwareUnitOfWork struct {
    *BasicUnitOfWork
    eventPublisher EventPublisher
    events         []DomainEvent
}

func (eauow *EventAwareUnitOfWork) RegisterEvent(event DomainEvent) {
    eauow.events = append(eauow.events, event)
}

func (eauow *EventAwareUnitOfWork) Commit(ctx context.Context) error {
    if err := eauow.BasicUnitOfWork.Commit(ctx); err != nil {
        return err
    }

    // Publish events after successful commit
    for _, event := range eauow.events {
        eauow.eventPublisher.Publish(event)
    }

    return nil
}
```

### 3. Saga Pattern Integration

```go
type SagaUnitOfWork struct {
    *BasicUnitOfWork
    compensations []CompensationAction
}

type CompensationAction func() error

func (suow *SagaUnitOfWork) RegisterCompensation(action CompensationAction) {
    suow.compensations = append(suow.compensations, action)
}

func (suow *SagaUnitOfWork) rollback(cause error) error {
    // Execute compensations in reverse order
    for i := len(suow.compensations) - 1; i >= 0; i-- {
        if err := suow.compensations[i](); err != nil {
            suow.logger.Error("Compensation failed", zap.Error(err))
        }
    }

    return suow.BasicUnitOfWork.rollback(cause)
}
```

## Common Interview Questions

### 1. **How does Unit of Work differ from Repository pattern?**

**Answer:**

| Aspect          | Repository              | Unit of Work              |
| --------------- | ----------------------- | ------------------------- |
| **Scope**       | Single entity type      | Multiple entity types     |
| **Purpose**     | Data access abstraction | Transaction coordination  |
| **Lifecycle**   | Per entity operation    | Per business transaction  |
| **State**       | Stateless               | Stateful (tracks changes) |
| **Transaction** | Individual operations   | Batched operations        |

**Repository Example:**

```go
// Each repository handles one entity type
userRepo.Create(user)
orderRepo.Create(order)
productRepo.Update(product)
// Each call is separate transaction
```

**Unit of Work Example:**

```go
// Unit of Work coordinates multiple entities
uow.RegisterNew(user)
uow.RegisterNew(order)
uow.RegisterModified(product)
uow.Commit() // All in one transaction
```

**Integration:**

```go
type ServiceWithUnitOfWork struct {
    uow       UnitOfWork
    userRepo  Repository
    orderRepo Repository
}

func (s *ServiceWithUnitOfWork) ProcessOrder(user *User, order *Order) error {
    s.uow.RegisterNew(user)
    s.uow.RegisterNew(order)

    // Update inventory
    products := s.getOrderProducts(order)
    for _, product := range products {
        product.Stock--
        s.uow.RegisterModified(product)
    }

    return s.uow.Commit(context.Background())
}
```

### 2. **How do you handle concurrency in Unit of Work?**

**Answer:**

**1. Optimistic Locking:**

```go
type VersionedEntity struct {
    ID      string
    Version int64
    Data    interface{}
}

func (uow *UnitOfWork) Update(entity *VersionedEntity) error {
    current, err := uow.repo.FindByID(entity.ID)
    if err != nil {
        return err
    }

    if current.Version != entity.Version {
        return &OptimisticLockingError{
            EntityID:        entity.ID,
            ExpectedVersion: entity.Version,
            ActualVersion:   current.Version,
        }
    }

    entity.Version++
    uow.RegisterModified(entity)
    return nil
}
```

**2. Pessimistic Locking:**

```go
func (uow *UnitOfWork) LockAndUpdate(entityID string) (*Entity, error) {
    entity, err := uow.repo.FindByIDForUpdate(entityID) // SELECT FOR UPDATE
    if err != nil {
        return nil, err
    }

    // Entity is locked until transaction commits/rollbacks
    return entity, nil
}
```

**3. Isolation Levels:**

```go
type TransactionOptions struct {
    IsolationLevel string // READ_COMMITTED, REPEATABLE_READ, SERIALIZABLE
    Timeout        time.Duration
    ReadOnly       bool
}

func NewUnitOfWorkWithOptions(opts TransactionOptions) (*UnitOfWork, error) {
    tx, err := db.BeginTransactionWithOptions(opts)
    if err != nil {
        return nil, err
    }

    return &UnitOfWork{transaction: tx}, nil
}
```

**4. Conflict Resolution:**

```go
type ConflictResolver interface {
    ResolveConflict(local, remote Entity) (Entity, error)
}

type MergeConflictResolver struct{}

func (mcr *MergeConflictResolver) ResolveConflict(local, remote Entity) (Entity, error) {
    // Merge changes from both versions
    merged := local.Clone()

    // Apply non-conflicting changes from remote
    changes := calculateNonConflictingChanges(local, remote)
    applyChanges(merged, changes)

    return merged, nil
}
```

### 3. **How do you implement Unit of Work with event sourcing?**

**Answer:**

**Event Sourcing Unit of Work tracks events instead of entity state:**

```go
type EventSourcingUnitOfWork struct {
    events      []DomainEvent
    eventStore  EventStore
    snapshots   map[string]interface{}
    aggregates  map[string]Aggregate
    transaction Transaction
    logger      *zap.Logger
}

func (esuow *EventSourcingUnitOfWork) RegisterEvent(event DomainEvent) {
    esuow.events = append(esuow.events, event)

    // Update aggregate state
    aggregateID := event.GetAggregateID()
    if aggregate, exists := esuow.aggregates[aggregateID]; exists {
        aggregate.Apply(event)
    }
}

func (esuow *EventSourcingUnitOfWork) RegisterAggregate(aggregate Aggregate) {
    esuow.aggregates[aggregate.GetID()] = aggregate

    // Register all uncommitted events
    for _, event := range aggregate.GetUncommittedEvents() {
        esuow.RegisterEvent(event)
    }
}

func (esuow *EventSourcingUnitOfWork) Commit(ctx context.Context) error {
    if len(esuow.events) == 0 {
        return nil
    }

    // Store events atomically
    if err := esuow.eventStore.SaveEvents(ctx, esuow.events); err != nil {
        return fmt.Errorf("failed to save events: %w", err)
    }

    // Mark events as committed
    for _, aggregate := range esuow.aggregates {
        aggregate.MarkEventsAsCommitted()
    }

    esuow.logger.Info("Event sourcing unit of work committed",
        zap.Int("event_count", len(esuow.events)))

    return nil
}

// Aggregate interface for event sourcing
type Aggregate interface {
    GetID() string
    GetVersion() int64
    Apply(event DomainEvent)
    GetUncommittedEvents() []DomainEvent
    MarkEventsAsCommitted()
}

// Domain event interface
type DomainEvent interface {
    GetAggregateID() string
    GetEventType() string
    GetTimestamp() time.Time
    GetVersion() int64
}

// Example aggregate
type OrderAggregate struct {
    ID               string
    Status           string
    Items            []OrderItem
    Version          int64
    uncommittedEvents []DomainEvent
}

func (oa *OrderAggregate) AddItem(item OrderItem) {
    event := &ItemAddedEvent{
        AggregateID: oa.ID,
        Item:        item,
        Timestamp:   time.Now(),
        Version:     oa.Version + 1,
    }

    oa.Apply(event)
    oa.uncommittedEvents = append(oa.uncommittedEvents, event)
}

func (oa *OrderAggregate) Apply(event DomainEvent) {
    switch e := event.(type) {
    case *ItemAddedEvent:
        oa.Items = append(oa.Items, e.Item)
        oa.Version = e.GetVersion()
    case *OrderConfirmedEvent:
        oa.Status = "CONFIRMED"
        oa.Version = e.GetVersion()
    }
}

func (oa *OrderAggregate) GetUncommittedEvents() []DomainEvent {
    return oa.uncommittedEvents
}

func (oa *OrderAggregate) MarkEventsAsCommitted() {
    oa.uncommittedEvents = make([]DomainEvent, 0)
}
```

### 4. **How do you test Unit of Work implementations?**

**Answer:**

**1. Mock-based Testing:**

```go
func TestUnitOfWorkCommit(t *testing.T) {
    // Setup mocks
    mockTransaction := &MockTransaction{}
    mockUserRepo := &MockRepository{}
    mockOrderRepo := &MockRepository{}

    repositories := map[string]Repository{
        "user":  mockUserRepo,
        "order": mockOrderRepo,
    }

    uow := NewBasicUnitOfWork(repositories, mockTransaction, logger)

    // Create test data
    user := &User{ID: "user1", Name: "John", Version: 0}
    order := &Order{ID: "order1", UserID: "user1", Version: 0}

    // Test registration
    uow.RegisterNew(user)
    uow.RegisterNew(order)

    // Setup expectations
    mockUserRepo.On("Create", mock.Anything, user).Return(nil)
    mockOrderRepo.On("Create", mock.Anything, order).Return(nil)
    mockTransaction.On("Commit").Return(nil)

    // Execute
    err := uow.Commit(context.Background())

    // Verify
    assert.NoError(t, err)
    mockUserRepo.AssertExpectations(t)
    mockOrderRepo.AssertExpectations(t)
    mockTransaction.AssertExpectations(t)

    stats := uow.GetStatistics()
    assert.True(t, stats.IsCommitted)
    assert.False(t, stats.IsRolledBack)
}
```

**2. Integration Testing:**

```go
func TestUnitOfWorkIntegration(t *testing.T) {
    // Use real database with transaction
    db := setupTestDatabase(t)
    defer db.Close()

    tx, err := db.BeginTransaction()
    require.NoError(t, err)

    repositories := map[string]Repository{
        "user":  NewSQLUserRepository(db),
        "order": NewSQLOrderRepository(db),
    }

    uow := NewBasicUnitOfWork(repositories, tx, logger)

    // Test complete workflow
    user := &User{ID: "user1", Name: "John", Email: "john@test.com"}
    order := &Order{ID: "order1", UserID: user.ID, Total: decimal.NewFromInt(100)}

    uow.RegisterNew(user)
    uow.RegisterNew(order)

    err = uow.Commit(context.Background())
    require.NoError(t, err)

    // Verify data was actually saved
    savedUser, err := repositories["user"].FindByID(context.Background(), user.ID)
    require.NoError(t, err)
    assert.Equal(t, user.Name, savedUser.(*User).Name)
}
```

**3. Error Scenario Testing:**

```go
func TestUnitOfWorkRollback(t *testing.T) {
    mockTransaction := &MockTransaction{}
    mockRepo := &MockRepository{}

    repositories := map[string]Repository{"user": mockRepo}
    uow := NewBasicUnitOfWork(repositories, mockTransaction, logger)

    user := &User{ID: "user1", Name: "John"}
    uow.RegisterNew(user)

    // Simulate repository error
    mockRepo.On("Create", mock.Anything, user).Return(fmt.Errorf("database error"))
    mockTransaction.On("Rollback").Return(nil)

    err := uow.Commit(context.Background())

    assert.Error(t, err)
    assert.Contains(t, err.Error(), "database error")

    stats := uow.GetStatistics()
    assert.False(t, stats.IsCommitted)
    assert.True(t, stats.IsRolledBack)

    mockRepo.AssertExpectations(t)
    mockTransaction.AssertExpectations(t)
}
```

**4. Concurrency Testing:**

```go
func TestUnitOfWorkConcurrency(t *testing.T) {
    db := setupTestDatabase(t)
    defer db.Close()

    // Create initial entity
    user := &User{ID: "user1", Name: "John", Version: 0}
    db.CreateUser(user)

    // Test concurrent modifications
    var wg sync.WaitGroup
    errors := make(chan error, 2)

    for i := 0; i < 2; i++ {
        wg.Add(1)
        go func(iteration int) {
            defer wg.Done()

            tx, _ := db.BeginTransaction()
            uow := NewBasicUnitOfWork(map[string]Repository{
                "user": NewSQLUserRepository(db),
            }, tx, logger)

            // Both goroutines try to update same user
            user.Name = fmt.Sprintf("John-%d", iteration)
            uow.RegisterModified(user)

            err := uow.Commit(context.Background())
            errors <- err
        }(i)
    }

    wg.Wait()
    close(errors)

    // One should succeed, one should fail with optimistic locking error
    successCount := 0
    lockingErrorCount := 0

    for err := range errors {
        if err == nil {
            successCount++
        } else if isOptimisticLockingError(err) {
            lockingErrorCount++
        }
    }

    assert.Equal(t, 1, successCount)
    assert.Equal(t, 1, lockingErrorCount)
}
```

### 5. **When should you avoid using Unit of Work?**

**Answer:**

**Avoid Unit of Work when:**

**1. Simple CRUD Operations:**

```go
// Don't use UoW for simple operations
func (s *UserService) CreateUser(user *User) error {
    // Simple create - no need for UoW overhead
    return s.userRepo.Create(context.Background(), user)
}

// Use UoW for complex operations
func (s *OrderService) ProcessOrder(order *Order) error {
    uow := s.uowFactory.Create()

    // Multiple related operations
    uow.RegisterNew(order)
    uow.RegisterModified(order.Customer)

    for _, item := range order.Items {
        product, _ := s.productRepo.FindByID(item.ProductID)
        product.Stock -= item.Quantity
        uow.RegisterModified(product)
    }

    return uow.Commit(context.Background())
}
```

**2. High-Performance Scenarios:**

```go
// High throughput - avoid UoW overhead
func (s *LogService) WriteLog(entry *LogEntry) error {
    // Direct write for performance
    return s.logRepo.Insert(entry)
}

// Use UoW for transactional integrity
func (s *PaymentService) ProcessPayment(payment *Payment) error {
    // Need atomicity across multiple tables
    uow := s.uowFactory.Create()
    // ... complex transaction logic
    return uow.Commit(context.Background())
}
```

**3. Event-Driven Architectures:**

```go
// Event handlers often don't need UoW
func (h *EventHandler) HandleUserCreated(event UserCreatedEvent) error {
    // Simple event processing
    return h.notificationService.SendWelcomeEmail(event.UserID)
}

// Use UoW for event sourcing
func (h *SagaHandler) HandlePaymentEvent(event PaymentEvent) error {
    uow := h.uowFactory.Create()
    // Complex saga orchestration
    return uow.Commit(context.Background())
}
```

**4. Microservices with Distributed Transactions:**

```go
// Don't use local UoW for distributed operations
func (s *DistributedOrderService) ProcessOrder(order *Order) error {
    // Use distributed transaction patterns instead
    saga := s.sagaManager.StartSaga("order-processing")

    // Step 1: Reserve inventory (different service)
    saga.Execute("reserve-inventory", order.Items)

    // Step 2: Process payment (different service)
    saga.Execute("process-payment", order.Payment)

    // Step 3: Create order (local service)
    saga.Execute("create-order", order)

    return saga.Complete()
}
```

**Decision Matrix:**

```go
type UnitOfWorkDecision struct {
    OperationComplexity  string // "simple", "complex"
    TransactionalNeeds   bool
    PerformanceRequirements string // "low", "high"
    EntityCount          int
    ConcurrencyLevel     string // "low", "high"
    DistributedSystem    bool
}

func (uowd *UnitOfWorkDecision) ShouldUseUnitOfWork() bool {
    if uowd.DistributedSystem {
        return false // Use distributed transaction patterns
    }

    if uowd.PerformanceRequirements == "high" && uowd.OperationComplexity == "simple" {
        return false // Overhead not justified
    }

    if uowd.EntityCount <= 1 && !uowd.TransactionalNeeds {
        return false // Single entity operations
    }

    if uowd.OperationComplexity == "complex" || uowd.TransactionalNeeds {
        return true // Complex operations benefit from UoW
    }

    return false
}
```
