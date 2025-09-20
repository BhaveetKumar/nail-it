# Iterator Pattern

## Pattern Name & Intent

**Iterator** is a behavioral design pattern that lets you traverse elements of a collection without exposing its underlying representation (list, stack, tree, etc.). It provides a way to access the elements of an aggregate object sequentially without exposing its internal structure.

**Key Intent:**

- Provide a uniform interface for traversing different types of collections
- Hide the internal structure and implementation of collections
- Support multiple concurrent iterations over the same collection
- Enable different traversal algorithms for the same collection
- Separate iteration logic from the collection implementation
- Support lazy evaluation and filtering during iteration

## When to Use

**Use Iterator when:**

1. **Collection Abstraction**: Need to traverse collections without exposing internal structure
2. **Multiple Traversals**: Support different ways of traversing the same collection
3. **Uniform Interface**: Want a consistent interface across different collection types
4. **Lazy Loading**: Need to load/compute elements on-demand during iteration
5. **Memory Efficiency**: Working with large datasets that don't fit in memory
6. **Filtering/Transformation**: Want to apply filters or transformations during iteration
7. **Concurrent Access**: Multiple clients need to iterate simultaneously

**Don't use when:**

- Simple collections with direct access methods
- Collection size is always small and fits in memory
- Only one type of iteration is needed
- Performance overhead is unacceptable
- Collection structure is very simple (e.g., single array)

## Real-World Use Cases (Payments/Fintech)

### 1. Transaction History Iterator

```go
// Transaction represents a financial transaction
type Transaction struct {
    ID          string
    Amount      decimal.Decimal
    Currency    string
    Type        string // CREDIT, DEBIT, TRANSFER
    Status      string // PENDING, COMPLETED, FAILED
    Timestamp   time.Time
    AccountID   string
    Description string
    Metadata    map[string]interface{}
}

// TransactionIterator interface for iterating through transactions
type TransactionIterator interface {
    HasNext() bool
    Next() (*Transaction, error)
    Reset()
    Count() int
    Close() error
}

// TransactionFilter defines filtering criteria
type TransactionFilter struct {
    AccountID    string
    Type         string
    Status       string
    MinAmount    decimal.Decimal
    MaxAmount    decimal.Decimal
    StartDate    time.Time
    EndDate      time.Time
    Currency     string
}

// Database-backed transaction iterator for large datasets
type DatabaseTransactionIterator struct {
    db           Database
    filter       *TransactionFilter
    batchSize    int
    currentBatch []*Transaction
    batchIndex   int
    totalCount   int
    hasMore      bool
    offset       int
    logger       *zap.Logger
}

func NewDatabaseTransactionIterator(db Database, filter *TransactionFilter, batchSize int, logger *zap.Logger) (*DatabaseTransactionIterator, error) {
    iterator := &DatabaseTransactionIterator{
        db:        db,
        filter:    filter,
        batchSize: batchSize,
        hasMore:   true,
        logger:    logger,
    }

    // Get total count for the filter
    count, err := iterator.getTotalCount()
    if err != nil {
        return nil, fmt.Errorf("failed to get total count: %w", err)
    }
    iterator.totalCount = count

    // Load first batch
    if err := iterator.loadNextBatch(); err != nil {
        return nil, fmt.Errorf("failed to load initial batch: %w", err)
    }

    return iterator, nil
}

func (d *DatabaseTransactionIterator) HasNext() bool {
    return d.batchIndex < len(d.currentBatch) || d.hasMore
}

func (d *DatabaseTransactionIterator) Next() (*Transaction, error) {
    // Load next batch if current batch is exhausted
    if d.batchIndex >= len(d.currentBatch) {
        if !d.hasMore {
            return nil, fmt.Errorf("no more transactions")
        }

        if err := d.loadNextBatch(); err != nil {
            return nil, fmt.Errorf("failed to load next batch: %w", err)
        }

        if len(d.currentBatch) == 0 {
            return nil, fmt.Errorf("no more transactions")
        }
    }

    transaction := d.currentBatch[d.batchIndex]
    d.batchIndex++

    return transaction, nil
}

func (d *DatabaseTransactionIterator) loadNextBatch() error {
    query := d.buildQuery()
    args := d.buildQueryArgs()

    d.logger.Debug("Loading transaction batch",
        zap.Int("offset", d.offset),
        zap.Int("batch_size", d.batchSize))

    rows, err := d.db.Query(query, args...)
    if err != nil {
        return fmt.Errorf("database query failed: %w", err)
    }
    defer rows.Close()

    var batch []*Transaction
    for rows.Next() {
        transaction := &Transaction{}
        err := rows.Scan(
            &transaction.ID,
            &transaction.Amount,
            &transaction.Currency,
            &transaction.Type,
            &transaction.Status,
            &transaction.Timestamp,
            &transaction.AccountID,
            &transaction.Description,
        )
        if err != nil {
            return fmt.Errorf("failed to scan transaction: %w", err)
        }

        batch = append(batch, transaction)
    }

    d.currentBatch = batch
    d.batchIndex = 0
    d.offset += len(batch)
    d.hasMore = len(batch) == d.batchSize

    d.logger.Debug("Loaded transaction batch",
        zap.Int("batch_size", len(batch)),
        zap.Bool("has_more", d.hasMore))

    return nil
}

func (d *DatabaseTransactionIterator) buildQuery() string {
    query := `
        SELECT id, amount, currency, type, status, timestamp, account_id, description
        FROM transactions
        WHERE 1=1
    `

    if d.filter.AccountID != "" {
        query += " AND account_id = ?"
    }
    if d.filter.Type != "" {
        query += " AND type = ?"
    }
    if d.filter.Status != "" {
        query += " AND status = ?"
    }
    if !d.filter.MinAmount.IsZero() {
        query += " AND amount >= ?"
    }
    if !d.filter.MaxAmount.IsZero() {
        query += " AND amount <= ?"
    }
    if !d.filter.StartDate.IsZero() {
        query += " AND timestamp >= ?"
    }
    if !d.filter.EndDate.IsZero() {
        query += " AND timestamp <= ?"
    }
    if d.filter.Currency != "" {
        query += " AND currency = ?"
    }

    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"

    return query
}

func (d *DatabaseTransactionIterator) buildQueryArgs() []interface{} {
    var args []interface{}

    if d.filter.AccountID != "" {
        args = append(args, d.filter.AccountID)
    }
    if d.filter.Type != "" {
        args = append(args, d.filter.Type)
    }
    if d.filter.Status != "" {
        args = append(args, d.filter.Status)
    }
    if !d.filter.MinAmount.IsZero() {
        args = append(args, d.filter.MinAmount)
    }
    if !d.filter.MaxAmount.IsZero() {
        args = append(args, d.filter.MaxAmount)
    }
    if !d.filter.StartDate.IsZero() {
        args = append(args, d.filter.StartDate)
    }
    if !d.filter.EndDate.IsZero() {
        args = append(args, d.filter.EndDate)
    }
    if d.filter.Currency != "" {
        args = append(args, d.filter.Currency)
    }

    args = append(args, d.batchSize, d.offset)

    return args
}

func (d *DatabaseTransactionIterator) getTotalCount() (int, error) {
    query := "SELECT COUNT(*) FROM transactions WHERE 1=1"
    args := d.buildQueryArgs()

    // Remove LIMIT and OFFSET from args for count query
    args = args[:len(args)-2]

    var count int
    err := d.db.QueryRow(query, args...).Scan(&count)
    if err != nil {
        return 0, err
    }

    return count, nil
}

func (d *DatabaseTransactionIterator) Reset() {
    d.offset = 0
    d.batchIndex = 0
    d.hasMore = true
    d.currentBatch = nil
}

func (d *DatabaseTransactionIterator) Count() int {
    return d.totalCount
}

func (d *DatabaseTransactionIterator) Close() error {
    d.currentBatch = nil
    return nil
}

// Filtered transaction iterator wrapper
type FilteredTransactionIterator struct {
    baseIterator TransactionIterator
    filter       func(*Transaction) bool
    prefetched   *Transaction
    logger       *zap.Logger
}

func NewFilteredTransactionIterator(baseIterator TransactionIterator, filter func(*Transaction) bool, logger *zap.Logger) *FilteredTransactionIterator {
    return &FilteredTransactionIterator{
        baseIterator: baseIterator,
        filter:       filter,
        logger:       logger,
    }
}

func (f *FilteredTransactionIterator) HasNext() bool {
    if f.prefetched != nil {
        return true
    }

    return f.findNext()
}

func (f *FilteredTransactionIterator) Next() (*Transaction, error) {
    if f.prefetched != nil {
        transaction := f.prefetched
        f.prefetched = nil
        return transaction, nil
    }

    if !f.findNext() {
        return nil, fmt.Errorf("no more matching transactions")
    }

    transaction := f.prefetched
    f.prefetched = nil
    return transaction, nil
}

func (f *FilteredTransactionIterator) findNext() bool {
    for f.baseIterator.HasNext() {
        transaction, err := f.baseIterator.Next()
        if err != nil {
            f.logger.Error("Error getting next transaction", zap.Error(err))
            return false
        }

        if f.filter(transaction) {
            f.prefetched = transaction
            return true
        }
    }

    return false
}

func (f *FilteredTransactionIterator) Reset() {
    f.baseIterator.Reset()
    f.prefetched = nil
}

func (f *FilteredTransactionIterator) Count() int {
    // Note: Count is not accurate for filtered iterators
    return f.baseIterator.Count()
}

func (f *FilteredTransactionIterator) Close() error {
    return f.baseIterator.Close()
}

// Transaction aggregator using iterator
type TransactionAggregator struct {
    logger *zap.Logger
}

func NewTransactionAggregator(logger *zap.Logger) *TransactionAggregator {
    return &TransactionAggregator{logger: logger}
}

func (ta *TransactionAggregator) CalculateTotals(iterator TransactionIterator) (*TransactionSummary, error) {
    summary := &TransactionSummary{
        TotalsByType:     make(map[string]decimal.Decimal),
        TotalsByCurrency: make(map[string]decimal.Decimal),
        TotalsByStatus:   make(map[string]decimal.Decimal),
    }

    defer iterator.Close()

    for iterator.HasNext() {
        transaction, err := iterator.Next()
        if err != nil {
            return nil, fmt.Errorf("failed to get next transaction: %w", err)
        }

        summary.TotalCount++
        summary.TotalAmount = summary.TotalAmount.Add(transaction.Amount)

        // Aggregate by type
        if total, exists := summary.TotalsByType[transaction.Type]; exists {
            summary.TotalsByType[transaction.Type] = total.Add(transaction.Amount)
        } else {
            summary.TotalsByType[transaction.Type] = transaction.Amount
        }

        // Aggregate by currency
        if total, exists := summary.TotalsByCurrency[transaction.Currency]; exists {
            summary.TotalsByCurrency[transaction.Currency] = total.Add(transaction.Amount)
        } else {
            summary.TotalsByCurrency[transaction.Currency] = transaction.Amount
        }

        // Aggregate by status
        if total, exists := summary.TotalsByStatus[transaction.Status]; exists {
            summary.TotalsByStatus[transaction.Status] = total.Add(transaction.Amount)
        } else {
            summary.TotalsByStatus[transaction.Status] = transaction.Amount
        }

        // Track date range
        if summary.EarliestTransaction.IsZero() || transaction.Timestamp.Before(summary.EarliestTransaction) {
            summary.EarliestTransaction = transaction.Timestamp
        }
        if summary.LatestTransaction.IsZero() || transaction.Timestamp.After(summary.LatestTransaction) {
            summary.LatestTransaction = transaction.Timestamp
        }
    }

    ta.logger.Info("Transaction aggregation completed",
        zap.Int("total_count", summary.TotalCount),
        zap.String("total_amount", summary.TotalAmount.String()))

    return summary, nil
}

type TransactionSummary struct {
    TotalCount           int
    TotalAmount          decimal.Decimal
    TotalsByType         map[string]decimal.Decimal
    TotalsByCurrency     map[string]decimal.Decimal
    TotalsByStatus       map[string]decimal.Decimal
    EarliestTransaction  time.Time
    LatestTransaction    time.Time
}
```

### 2. Account Balance History Iterator

```go
// BalanceEntry represents a point-in-time balance
type BalanceEntry struct {
    AccountID   string
    Balance     decimal.Decimal
    Currency    string
    Timestamp   time.Time
    ChangeType  string // CREDIT, DEBIT, ADJUSTMENT
    Amount      decimal.Decimal
    Description string
    Reference   string
}

// Time-series balance iterator
type BalanceHistoryIterator interface {
    HasNext() bool
    Next() (*BalanceEntry, error)
    Peek() (*BalanceEntry, error)
    Skip(n int) error
    Reset()
    Close() error

    // Time-series specific methods
    SeekToTime(timestamp time.Time) error
    GetTimeRange() (start, end time.Time)
}

// Memory-efficient balance history iterator
type StreamingBalanceIterator struct {
    accountID    string
    db           Database
    startTime    time.Time
    endTime      time.Time
    currentTime  time.Time
    batchSize    int
    currentBatch []*BalanceEntry
    batchIndex   int
    hasMore      bool
    logger       *zap.Logger
}

func NewStreamingBalanceIterator(
    db Database,
    accountID string,
    startTime, endTime time.Time,
    batchSize int,
    logger *zap.Logger,
) *StreamingBalanceIterator {
    return &StreamingBalanceIterator{
        accountID:   accountID,
        db:          db,
        startTime:   startTime,
        endTime:     endTime,
        currentTime: startTime,
        batchSize:   batchSize,
        hasMore:     true,
        logger:      logger,
    }
}

func (s *StreamingBalanceIterator) HasNext() bool {
    return s.batchIndex < len(s.currentBatch) || s.hasMore
}

func (s *StreamingBalanceIterator) Next() (*BalanceEntry, error) {
    if s.batchIndex >= len(s.currentBatch) {
        if !s.hasMore {
            return nil, fmt.Errorf("no more balance entries")
        }

        if err := s.loadNextBatch(); err != nil {
            return nil, err
        }

        if len(s.currentBatch) == 0 {
            return nil, fmt.Errorf("no more balance entries")
        }
    }

    entry := s.currentBatch[s.batchIndex]
    s.batchIndex++

    return entry, nil
}

func (s *StreamingBalanceIterator) Peek() (*BalanceEntry, error) {
    if s.batchIndex >= len(s.currentBatch) {
        if !s.hasMore {
            return nil, fmt.Errorf("no more balance entries")
        }

        if err := s.loadNextBatch(); err != nil {
            return nil, err
        }

        if len(s.currentBatch) == 0 {
            return nil, fmt.Errorf("no more balance entries")
        }
    }

    return s.currentBatch[s.batchIndex], nil
}

func (s *StreamingBalanceIterator) Skip(n int) error {
    for i := 0; i < n && s.HasNext(); i++ {
        _, err := s.Next()
        if err != nil {
            return err
        }
    }
    return nil
}

func (s *StreamingBalanceIterator) loadNextBatch() error {
    query := `
        SELECT account_id, balance, currency, timestamp, change_type, amount, description, reference
        FROM balance_history
        WHERE account_id = ? AND timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp ASC
        LIMIT ?
    `

    rows, err := s.db.Query(query, s.accountID, s.currentTime, s.endTime, s.batchSize)
    if err != nil {
        return fmt.Errorf("failed to query balance history: %w", err)
    }
    defer rows.Close()

    var batch []*BalanceEntry
    for rows.Next() {
        entry := &BalanceEntry{}
        err := rows.Scan(
            &entry.AccountID,
            &entry.Balance,
            &entry.Currency,
            &entry.Timestamp,
            &entry.ChangeType,
            &entry.Amount,
            &entry.Description,
            &entry.Reference,
        )
        if err != nil {
            return fmt.Errorf("failed to scan balance entry: %w", err)
        }

        batch = append(batch, entry)
        s.currentTime = entry.Timestamp.Add(time.Nanosecond) // Move past this entry
    }

    s.currentBatch = batch
    s.batchIndex = 0
    s.hasMore = len(batch) == s.batchSize

    return nil
}

func (s *StreamingBalanceIterator) SeekToTime(timestamp time.Time) error {
    if timestamp.Before(s.startTime) || timestamp.After(s.endTime) {
        return fmt.Errorf("timestamp outside iterator range")
    }

    s.currentTime = timestamp
    s.batchIndex = 0
    s.currentBatch = nil
    s.hasMore = true

    return s.loadNextBatch()
}

func (s *StreamingBalanceIterator) GetTimeRange() (start, end time.Time) {
    return s.startTime, s.endTime
}

func (s *StreamingBalanceIterator) Reset() {
    s.currentTime = s.startTime
    s.batchIndex = 0
    s.currentBatch = nil
    s.hasMore = true
}

func (s *StreamingBalanceIterator) Close() error {
    s.currentBatch = nil
    return nil
}

// Balance calculator using iterator
type BalanceCalculator struct {
    logger *zap.Logger
}

func NewBalanceCalculator(logger *zap.Logger) *BalanceCalculator {
    return &BalanceCalculator{logger: logger}
}

func (bc *BalanceCalculator) CalculateAverageBalance(iterator BalanceHistoryIterator) (decimal.Decimal, error) {
    defer iterator.Close()

    var totalBalance decimal.Decimal
    var count int

    for iterator.HasNext() {
        entry, err := iterator.Next()
        if err != nil {
            return decimal.Zero, fmt.Errorf("failed to get next balance entry: %w", err)
        }

        totalBalance = totalBalance.Add(entry.Balance)
        count++
    }

    if count == 0 {
        return decimal.Zero, fmt.Errorf("no balance entries found")
    }

    average := totalBalance.Div(decimal.NewFromInt(int64(count)))

    bc.logger.Info("Average balance calculated",
        zap.String("average", average.String()),
        zap.Int("entries", count))

    return average, nil
}

func (bc *BalanceCalculator) CalculateMinMaxBalance(iterator BalanceHistoryIterator) (min, max decimal.Decimal, err error) {
    defer iterator.Close()

    var initialized bool

    for iterator.HasNext() {
        entry, err := iterator.Next()
        if err != nil {
            return decimal.Zero, decimal.Zero, fmt.Errorf("failed to get next balance entry: %w", err)
        }

        if !initialized {
            min = entry.Balance
            max = entry.Balance
            initialized = true
        } else {
            if entry.Balance.LessThan(min) {
                min = entry.Balance
            }
            if entry.Balance.GreaterThan(max) {
                max = entry.Balance
            }
        }
    }

    if !initialized {
        return decimal.Zero, decimal.Zero, fmt.Errorf("no balance entries found")
    }

    return min, max, nil
}
```

### 3. Payment Batch Iterator

```go
// PaymentBatch represents a batch of payments to be processed
type PaymentBatch struct {
    ID          string
    Payments    []*Payment
    Status      string
    CreatedAt   time.Time
    ProcessedAt *time.Time
    TotalAmount decimal.Decimal
    Currency    string
    Metadata    map[string]interface{}
}

// Payment represents an individual payment
type Payment struct {
    ID            string
    Amount        decimal.Decimal
    Currency      string
    FromAccountID string
    ToAccountID   string
    Description   string
    Status        string
    CreatedAt     time.Time
    ProcessedAt   *time.Time
    ErrorMessage  string
}

// PaymentBatchIterator for processing large payment batches
type PaymentBatchIterator interface {
    HasNext() bool
    Next() (*Payment, error)
    GetBatchInfo() *PaymentBatch
    GetProgress() (current, total int)
    Reset()
    Close() error
}

// Concurrent payment batch iterator
type ConcurrentPaymentBatchIterator struct {
    batch           *PaymentBatch
    currentIndex    int32
    paymentChannel  chan *Payment
    errorChannel    chan error
    done            chan struct{}
    mu              sync.RWMutex
    closed          bool
    logger          *zap.Logger
}

func NewConcurrentPaymentBatchIterator(batch *PaymentBatch, bufferSize int, logger *zap.Logger) *ConcurrentPaymentBatchIterator {
    iterator := &ConcurrentPaymentBatchIterator{
        batch:          batch,
        paymentChannel: make(chan *Payment, bufferSize),
        errorChannel:   make(chan error, 1),
        done:           make(chan struct{}),
        logger:         logger,
    }

    // Start goroutine to feed payments into channel
    go iterator.feedPayments()

    return iterator
}

func (c *ConcurrentPaymentBatchIterator) feedPayments() {
    defer close(c.paymentChannel)
    defer close(c.errorChannel)

    for _, payment := range c.batch.Payments {
        select {
        case c.paymentChannel <- payment:
            atomic.AddInt32(&c.currentIndex, 1)
        case <-c.done:
            return
        }
    }
}

func (c *ConcurrentPaymentBatchIterator) HasNext() bool {
    c.mu.RLock()
    defer c.mu.RUnlock()

    if c.closed {
        return false
    }

    // Check if there are more payments in the channel
    select {
    case <-c.paymentChannel:
        // Put it back (this is a peek operation)
        return true
    default:
        return int(atomic.LoadInt32(&c.currentIndex)) < len(c.batch.Payments)
    }
}

func (c *ConcurrentPaymentBatchIterator) Next() (*Payment, error) {
    c.mu.RLock()
    defer c.mu.RUnlock()

    if c.closed {
        return nil, fmt.Errorf("iterator is closed")
    }

    select {
    case payment, ok := <-c.paymentChannel:
        if !ok {
            return nil, fmt.Errorf("no more payments")
        }
        return payment, nil
    case err := <-c.errorChannel:
        return nil, err
    case <-c.done:
        return nil, fmt.Errorf("iterator closed")
    }
}

func (c *ConcurrentPaymentBatchIterator) GetBatchInfo() *PaymentBatch {
    return c.batch
}

func (c *ConcurrentPaymentBatchIterator) GetProgress() (current, total int) {
    return int(atomic.LoadInt32(&c.currentIndex)), len(c.batch.Payments)
}

func (c *ConcurrentPaymentBatchIterator) Reset() {
    c.mu.Lock()
    defer c.mu.Unlock()

    if !c.closed {
        close(c.done)
        c.closed = true
    }

    // Create new iterator state
    c.currentIndex = 0
    c.paymentChannel = make(chan *Payment, cap(c.paymentChannel))
    c.errorChannel = make(chan error, 1)
    c.done = make(chan struct{})
    c.closed = false

    // Restart feeding
    go c.feedPayments()
}

func (c *ConcurrentPaymentBatchIterator) Close() error {
    c.mu.Lock()
    defer c.mu.Unlock()

    if !c.closed {
        close(c.done)
        c.closed = true
    }

    return nil
}

// Batch processor using iterator
type PaymentBatchProcessor struct {
    processor PaymentProcessor
    logger    *zap.Logger
}

type PaymentProcessor interface {
    ProcessPayment(ctx context.Context, payment *Payment) error
}

func NewPaymentBatchProcessor(processor PaymentProcessor, logger *zap.Logger) *PaymentBatchProcessor {
    return &PaymentBatchProcessor{
        processor: processor,
        logger:    logger,
    }
}

func (p *PaymentBatchProcessor) ProcessBatch(ctx context.Context, iterator PaymentBatchIterator) (*BatchProcessingResult, error) {
    result := &BatchProcessingResult{
        BatchID:      iterator.GetBatchInfo().ID,
        StartTime:    time.Now(),
        Processed:    make([]*Payment, 0),
        Failed:       make([]*Payment, 0),
        Errors:       make([]error, 0),
    }

    defer iterator.Close()

    p.logger.Info("Starting batch processing",
        zap.String("batch_id", result.BatchID))

    for iterator.HasNext() {
        payment, err := iterator.Next()
        if err != nil {
            p.logger.Error("Failed to get next payment", zap.Error(err))
            result.Errors = append(result.Errors, err)
            continue
        }

        if err := p.processor.ProcessPayment(ctx, payment); err != nil {
            p.logger.Error("Payment processing failed",
                zap.String("payment_id", payment.ID),
                zap.Error(err))

            payment.Status = "FAILED"
            payment.ErrorMessage = err.Error()
            result.Failed = append(result.Failed, payment)
            result.Errors = append(result.Errors, err)
        } else {
            payment.Status = "COMPLETED"
            now := time.Now()
            payment.ProcessedAt = &now
            result.Processed = append(result.Processed, payment)
        }

        result.TotalProcessed++
    }

    result.EndTime = time.Now()
    result.Duration = result.EndTime.Sub(result.StartTime)

    p.logger.Info("Batch processing completed",
        zap.String("batch_id", result.BatchID),
        zap.Int("total_processed", result.TotalProcessed),
        zap.Int("successful", len(result.Processed)),
        zap.Int("failed", len(result.Failed)),
        zap.Duration("duration", result.Duration))

    return result, nil
}

type BatchProcessingResult struct {
    BatchID        string
    StartTime      time.Time
    EndTime        time.Time
    Duration       time.Duration
    TotalProcessed int
    Processed      []*Payment
    Failed         []*Payment
    Errors         []error
}
```

## Go Implementation

```go
package main

import (
    "fmt"
    "time"
    "context"
    "github.com/shopspring/decimal"
    "go.uber.org/zap"
)

// Generic Iterator interface
type Iterator[T any] interface {
    HasNext() bool
    Next() (T, error)
    Reset()
}

// Collection interface
type Collection[T any] interface {
    CreateIterator() Iterator[T]
    Size() int
    Add(item T) error
    Remove(item T) error
    Contains(item T) bool
}

// Concrete implementation: Slice-based collection
type SliceCollection[T any] struct {
    items []T
}

func NewSliceCollection[T any]() *SliceCollection[T] {
    return &SliceCollection[T]{
        items: make([]T, 0),
    }
}

func (s *SliceCollection[T]) CreateIterator() Iterator[T] {
    return NewSliceIterator(s.items)
}

func (s *SliceCollection[T]) Size() int {
    return len(s.items)
}

func (s *SliceCollection[T]) Add(item T) error {
    s.items = append(s.items, item)
    return nil
}

func (s *SliceCollection[T]) Remove(item T) error {
    for i, existing := range s.items {
        // This is a simplified comparison; in real code you'd need proper equality
        if fmt.Sprintf("%v", existing) == fmt.Sprintf("%v", item) {
            s.items = append(s.items[:i], s.items[i+1:]...)
            return nil
        }
    }
    return fmt.Errorf("item not found")
}

func (s *SliceCollection[T]) Contains(item T) bool {
    for _, existing := range s.items {
        if fmt.Sprintf("%v", existing) == fmt.Sprintf("%v", item) {
            return true
        }
    }
    return false
}

// Slice iterator implementation
type SliceIterator[T any] struct {
    items   []T
    current int
}

func NewSliceIterator[T any](items []T/) *SliceIterator[T] {
    return &SliceIterator[T]{
        items:   items,
        current: 0,
    }
}

func (s *SliceIterator[T]) HasNext() bool {
    return s.current < len(s.items)
}

func (s *SliceIterator[T]) Next() (T, error) {
    var zero T
    if !s.HasNext() {
        return zero, fmt.Errorf("no more items")
    }

    item := s.items[s.current]
    s.current++
    return item, nil
}

func (s *SliceIterator[T]) Reset() {
    s.current = 0
}

// Range iterator for generating sequences
type RangeIterator struct {
    start   int
    end     int
    step    int
    current int
}

func NewRangeIterator(start, end, step int) *RangeIterator {
    return &RangeIterator{
        start:   start,
        end:     end,
        step:    step,
        current: start,
    }
}

func (r *RangeIterator) HasNext() bool {
    if r.step > 0 {
        return r.current < r.end
    }
    return r.current > r.end
}

func (r *RangeIterator) Next() (int, error) {
    if !r.HasNext() {
        return 0, fmt.Errorf("no more items")
    }

    value := r.current
    r.current += r.step
    return value, nil
}

func (r *RangeIterator) Reset() {
    r.current = r.start
}

// Filtered iterator wrapper
type FilteredIterator[T any] struct {
    baseIterator Iterator[T]
    filter       func(T) bool
    prefetched   *T
    hasPrefetch  bool
}

func NewFilteredIterator[T any](baseIterator Iterator[T], filter func(T/) bool) *FilteredIterator[T] {
    return &FilteredIterator[T]{
        baseIterator: baseIterator,
        filter:       filter,
    }
}

func (f *FilteredIterator[T]) HasNext() bool {
    if f.hasPrefetch {
        return true
    }

    return f.findNext()
}

func (f *FilteredIterator[T]) Next() (T, error) {
    var zero T

    if f.hasPrefetch {
        item := *f.prefetched
        f.prefetched = nil
        f.hasPrefetch = false
        return item, nil
    }

    if !f.findNext() {
        return zero, fmt.Errorf("no more matching items")
    }

    item := *f.prefetched
    f.prefetched = nil
    f.hasPrefetch = false
    return item, nil
}

func (f *FilteredIterator[T]) findNext() bool {
    for f.baseIterator.HasNext() {
        item, err := f.baseIterator.Next()
        if err != nil {
            return false
        }

        if f.filter(item) {
            f.prefetched = &item
            f.hasPrefetch = true
            return true
        }
    }

    return false
}

func (f *FilteredIterator[T]) Reset() {
    f.baseIterator.Reset()
    f.prefetched = nil
    f.hasPrefetch = false
}

// Transformed iterator
type TransformedIterator[T, U any] struct {
    baseIterator Iterator[T]
    transform    func(T) U
}

func NewTransformedIterator[T, U any](baseIterator Iterator[T], transform func(T/) U) *TransformedIterator[T, U] {
    return &TransformedIterator[T, U]{
        baseIterator: baseIterator,
        transform:    transform,
    }
}

func (t *TransformedIterator[T, U]) HasNext() bool {
    return t.baseIterator.HasNext()
}

func (t *TransformedIterator[T, U]) Next() (U, error) {
    var zero U

    item, err := t.baseIterator.Next()
    if err != nil {
        return zero, err
    }

    return t.transform(item), nil
}

func (t *TransformedIterator[T, U]) Reset() {
    t.baseIterator.Reset()
}

// Batched iterator for processing items in batches
type BatchedIterator[T any] struct {
    baseIterator Iterator[T]
    batchSize    int
}

func NewBatchedIterator[T any](baseIterator Iterator[T], batchSize int/) *BatchedIterator[T] {
    return &BatchedIterator[T]{
        baseIterator: baseIterator,
        batchSize:    batchSize,
    }
}

func (b *BatchedIterator[T]) HasNext() bool {
    return b.baseIterator.HasNext()
}

func (b *BatchedIterator[T]) Next() ([]T, error) {
    var batch []T

    for i := 0; i < b.batchSize && b.baseIterator.HasNext(); i++ {
        item, err := b.baseIterator.Next()
        if err != nil {
            return batch, err
        }
        batch = append(batch, item)
    }

    if len(batch) == 0 {
        return nil, fmt.Errorf("no more items")
    }

    return batch, nil
}

func (b *BatchedIterator[T]) Reset() {
    b.baseIterator.Reset()
}

// Chained iterator for combining multiple iterators
type ChainedIterator[T any] struct {
    iterators    []Iterator[T]
    currentIndex int
}

func NewChainedIterator[T any](iterators ...Iterator[T]/) *ChainedIterator[T] {
    return &ChainedIterator[T]{
        iterators:    iterators,
        currentIndex: 0,
    }
}

func (c *ChainedIterator[T]) HasNext() bool {
    for c.currentIndex < len(c.iterators) {
        if c.iterators[c.currentIndex].HasNext() {
            return true
        }
        c.currentIndex++
    }
    return false
}

func (c *ChainedIterator[T]) Next() (T, error) {
    var zero T

    if !c.HasNext() {
        return zero, fmt.Errorf("no more items")
    }

    return c.iterators[c.currentIndex].Next()
}

func (c *ChainedIterator[T]) Reset() {
    c.currentIndex = 0
    for _, iterator := range c.iterators {
        iterator.Reset()
    }
}

// Lazy iterator for on-demand generation
type LazyIterator[T any] struct {
    generator func() (T, bool)
    current   *T
    hasNext   bool
    checked   bool
}

func NewLazyIterator[T any](generator func(/) (T, bool)) *LazyIterator[T] {
    return &LazyIterator[T]{
        generator: generator,
    }
}

func (l *LazyIterator[T]) HasNext() bool {
    if !l.checked {
        item, hasNext := l.generator()
        l.current = &item
        l.hasNext = hasNext
        l.checked = true
    }
    return l.hasNext
}

func (l *LazyIterator[T]) Next() (T, error) {
    var zero T

    if !l.HasNext() {
        return zero, fmt.Errorf("no more items")
    }

    item := *l.current
    l.checked = false
    return item, nil
}

func (l *LazyIterator[T]) Reset() {
    l.current = nil
    l.hasNext = false
    l.checked = false
}

// Example: Book collection and processing
type Book struct {
    ID       string
    Title    string
    Author   string
    Genre    string
    Year     int
    Pages    int
    Rating   float64
    InStock  bool
}

func (b Book) String() string {
    return fmt.Sprintf("%s by %s (%d)", b.Title, b.Author, b.Year)
}

// Book processor using different iterators
type BookProcessor struct {
    logger *zap.Logger
}

func NewBookProcessor(logger *zap.Logger) *BookProcessor {
    return &BookProcessor{logger: logger}
}

func (bp *BookProcessor) ProcessBooks(iterator Iterator[Book]) {
    bp.logger.Info("Starting book processing")

    count := 0
    for iterator.HasNext() {
        book, err := iterator.Next()
        if err != nil {
            bp.logger.Error("Error getting next book", zap.Error(err))
            break
        }

        bp.logger.Debug("Processing book",
            zap.String("title", book.Title),
            zap.String("author", book.Author))

        count++
    }

    bp.logger.Info("Book processing completed", zap.Int("total_processed", count))
}

func (bp *BookProcessor) CalculateAverageRating(iterator Iterator[Book]) float64 {
    var totalRating float64
    var count int

    for iterator.HasNext() {
        book, err := iterator.Next()
        if err != nil {
            break
        }

        totalRating += book.Rating
        count++
    }

    if count == 0 {
        return 0
    }

    return totalRating / float64(count)
}

func (bp *BookProcessor) FindBooksByGenre(iterator Iterator[Book], genre string) []Book {
    var books []Book

    for iterator.HasNext() {
        book, err := iterator.Next()
        if err != nil {
            break
        }

        if book.Genre == genre {
            books = append(books, book)
        }
    }

    return books
}

// Utility functions for creating iterators
func FilterBooks(iterator Iterator[Book], predicate func(Book) bool) Iterator[Book] {
    return NewFilteredIterator(iterator, predicate)
}

func MapBooksToTitles(iterator Iterator[Book]) Iterator[string] {
    return NewTransformedIterator(iterator, func(book Book) string {
        return book.Title
    })
}

func BatchBooks(iterator Iterator[Book], batchSize int) Iterator[[]Book] {
    return NewBatchedIterator(iterator, batchSize)
}

// Example usage and demonstrations
func main() {
    fmt.Println("=== Iterator Pattern Demo ===\n")

    // Create logger
    logger, _ := zap.NewDevelopment()
    defer logger.Sync()

    // Create a collection of books
    books := NewSliceCollection[Book]()

    sampleBooks := []Book{
        {ID: "1", Title: "The Go Programming Language", Author: "Alan Donovan", Genre: "Technology", Year: 2015, Pages: 380, Rating: 4.5, InStock: true},
        {ID: "2", Title: "Clean Code", Author: "Robert Martin", Genre: "Technology", Year: 2008, Pages: 464, Rating: 4.7, InStock: true},
        {ID: "3", Title: "The Pragmatic Programmer", Author: "Andy Hunt", Genre: "Technology", Year: 1999, Pages: 352, Rating: 4.6, InStock: false},
        {ID: "4", Title: "Design Patterns", Author: "Gang of Four", Genre: "Technology", Year: 1994, Pages: 395, Rating: 4.4, InStock: true},
        {ID: "5", Title: "Dune", Author: "Frank Herbert", Genre: "Science Fiction", Year: 1965, Pages: 688, Rating: 4.8, InStock: true},
        {ID: "6", Title: "1984", Author: "George Orwell", Genre: "Dystopian", Year: 1949, Pages: 328, Rating: 4.9, InStock: true},
        {ID: "7", Title: "To Kill a Mockingbird", Author: "Harper Lee", Genre: "Fiction", Year: 1960, Pages: 281, Rating: 4.3, InStock: false},
    }

    for _, book := range sampleBooks {
        books.Add(book)
    }

    processor := NewBookProcessor(logger)

    // Basic iteration
    fmt.Println("=== Basic Iteration ===")
    basicIterator := books.CreateIterator()
    processor.ProcessBooks(basicIterator)

    // Filtered iteration - only technology books
    fmt.Println("\n=== Filtered Iteration (Technology Books) ===")
    techBooksIterator := FilterBooks(books.CreateIterator(), func(book Book) bool {
        return book.Genre == "Technology"
    })

    count := 0
    for techBooksIterator.HasNext() {
        book, err := techBooksIterator.Next()
        if err != nil {
            break
        }
        fmt.Printf("  %s\n", book)
        count++
    }
    fmt.Printf("Found %d technology books\n", count)

    // Transformed iteration - book titles only
    fmt.Println("\n=== Transformed Iteration (Titles Only) ===")
    titleIterator := MapBooksToTitles(books.CreateIterator())

    count = 0
    for titleIterator.HasNext() {
        title, err := titleIterator.Next()
        if err != nil {
            break
        }
        fmt.Printf("  %d. %s\n", count+1, title)
        count++
    }

    // Batched iteration
    fmt.Println("\n=== Batched Iteration (Batch Size 3) ===")
    batchIterator := BatchBooks(books.CreateIterator(), 3)

    batchNum := 1
    for batchIterator.HasNext() {
        batch, err := batchIterator.Next()
        if err != nil {
            break
        }

        fmt.Printf("Batch %d (%d books):\n", batchNum, len(batch))
        for _, book := range batch {
            fmt.Printf("  - %s\n", book.Title)
        }
        batchNum++
    }

    // Chained iteration - combining filtered results
    fmt.Println("\n=== Chained Iteration (Tech + Fiction) ===")
    techIterator := FilterBooks(books.CreateIterator(), func(book Book) bool {
        return book.Genre == "Technology"
    })

    fictionIterator := FilterBooks(books.CreateIterator(), func(book Book) bool {
        return book.Genre == "Fiction" || book.Genre == "Science Fiction"
    })

    chainedIterator := NewChainedIterator(techIterator, fictionIterator)

    count = 0
    for chainedIterator.HasNext() {
        book, err := chainedIterator.Next()
        if err != nil {
            break
        }
        fmt.Printf("  %s (%s)\n", book.Title, book.Genre)
        count++
    }
    fmt.Printf("Total books in chain: %d\n", count)

    // Range iterator
    fmt.Println("\n=== Range Iterator ===")
    rangeIterator := NewRangeIterator(1, 10, 2)

    fmt.Print("Odd numbers 1-9: ")
    for rangeIterator.HasNext() {
        num, err := rangeIterator.Next()
        if err != nil {
            break
        }
        fmt.Printf("%d ", num)
    }
    fmt.Println()

    // Lazy iterator - Fibonacci sequence
    fmt.Println("\n=== Lazy Iterator (Fibonacci) ===")
    fibIterator := createFibonacciIterator()

    fmt.Print("First 10 Fibonacci numbers: ")
    for i := 0; i < 10 && fibIterator.HasNext(); i++ {
        num, err := fibIterator.Next()
        if err != nil {
            break
        }
        fmt.Printf("%d ", num)
    }
    fmt.Println()

    // Calculate average rating using iterator
    fmt.Println("\n=== Calculate Average Rating ===")
    ratingIterator := books.CreateIterator()
    avgRating := processor.CalculateAverageRating(ratingIterator)
    fmt.Printf("Average book rating: %.2f\n", avgRating)

    // Find books by genre using iterator
    fmt.Println("\n=== Find Books by Genre ===")
    genreIterator := books.CreateIterator()
    techBooks := processor.FindBooksByGenre(genreIterator, "Technology")
    fmt.Printf("Found %d technology books:\n", len(techBooks))
    for _, book := range techBooks {
        fmt.Printf("  - %s (Rating: %.1f)\n", book.Title, book.Rating)
    }

    // Demonstrate iterator reset
    fmt.Println("\n=== Iterator Reset Demo ===")
    resetIterator := books.CreateIterator()

    fmt.Print("First 3 books: ")
    for i := 0; i < 3 && resetIterator.HasNext(); i++ {
        book, _ := resetIterator.Next()
        fmt.Printf("%s; ", book.Title)
    }
    fmt.Println()

    resetIterator.Reset()
    fmt.Print("After reset, first 3 again: ")
    for i := 0; i < 3 && resetIterator.HasNext(); i++ {
        book, _ := resetIterator.Next()
        fmt.Printf("%s; ", book.Title)
    }
    fmt.Println()

    fmt.Println("\n=== Iterator Pattern Demo Complete ===")
}

// Helper function to create Fibonacci iterator
func createFibonacciIterator() Iterator[int] {
    a, b := 0, 1

    generator := func() (int, bool) {
        current := a
        a, b = b, a+b
        return current, true // Infinite sequence
    }

    return NewLazyIterator(generator)
}
```

## Variants & Trade-offs

### Variants

1. **External Iterator (Active Iterator)**

```go
// Client controls the iteration
type ExternalIterator[T any] interface {
    HasNext() bool
    Next() (T, error)
    Reset()
}

func ProcessWithExternalIterator[T any](iterator ExternalIterator[T], processor func(T)/) {
    for iterator.HasNext() {
        item, err := iterator.Next()
        if err != nil {
            break
        }
        processor(item)
    }
}
```

2. **Internal Iterator (Passive Iterator)**

```go
// Collection controls the iteration
type InternalIterator[T any] interface {
    ForEach(func(T) bool) // bool indicates whether to continue
}

type SliceInternalIterator[T any] struct {
    items []T
}

func (s *SliceInternalIterator[T]) ForEach(fn func(T) bool) {
    for _, item := range s.items {
        if !fn(item) {
            break
        }
    }
}
```

3. **Bidirectional Iterator**

```go
type BidirectionalIterator[T any] interface {
    Iterator[T]
    HasPrevious() bool
    Previous() (T, error)
    First() (T, error)
    Last() (T, error)
}

type DoublyLinkedListIterator[T any] struct {
    list    *DoublyLinkedList[T]
    current *Node[T]
}

func (d *DoublyLinkedListIterator[T]) HasPrevious() bool {
    return d.current != nil && d.current.Prev != nil
}

func (d *DoublyLinkedListIterator[T]) Previous() (T, error) {
    var zero T
    if !d.HasPrevious() {
        return zero, fmt.Errorf("no previous item")
    }

    d.current = d.current.Prev
    return d.current.Data, nil
}
```

4. **Random Access Iterator**

```go
type RandomAccessIterator[T any] interface {
    BidirectionalIterator[T]
    Seek(index int) error
    GetIndex() int
    Distance(other RandomAccessIterator[T]) int
}

type ArrayIterator[T any] struct {
    array []T
    index int
}

func (a *ArrayIterator[T]) Seek(index int) error {
    if index < 0 || index >= len(a.array) {
        return fmt.Errorf("index out of bounds")
    }
    a.index = index
    return nil
}

func (a *ArrayIterator[T]) GetIndex() int {
    return a.index
}
```

### Trade-offs

**Pros:**

- **Abstraction**: Hides collection implementation details
- **Consistency**: Uniform interface across different collections
- **Memory Efficiency**: Can iterate without loading entire collection
- **Flexibility**: Support multiple traversal algorithms
- **Lazy Evaluation**: Compute elements on-demand
- **Concurrent Access**: Multiple iterators can traverse simultaneously

**Cons:**

- **Performance Overhead**: Iterator objects add memory and CPU overhead
- **Complexity**: More complex than direct array access
- **State Management**: Iterator state must be carefully managed
- **Error Handling**: Need to handle iteration errors appropriately
- **Thread Safety**: Concurrent modification can invalidate iterators

**When to Choose Iterator vs Alternatives:**

| Scenario            | Pattern          | Reason                     |
| ------------------- | ---------------- | -------------------------- |
| Large collections   | Iterator         | Memory efficiency          |
| Small arrays        | Direct access    | Less overhead              |
| Multiple traversals | Iterator         | Flexibility                |
| Single pass         | Direct iteration | Simplicity                 |
| Lazy evaluation     | Iterator         | On-demand computation      |
| Stream processing   | Pipeline         | Better for transformations |

## Integration Tips

### 1. Builder Pattern Integration

```go
type IteratorBuilder[T any] struct {
    baseIterator Iterator[T]
    filters      []func(T) bool
    transforms   []func(T) T
    batchSize    int
    limit        int
}

func NewIteratorBuilder[T any](baseIterator Iterator[T]/) *IteratorBuilder[T] {
    return &IteratorBuilder[T]{
        baseIterator: baseIterator,
    }
}

func (ib *IteratorBuilder[T]) Filter(predicate func(T) bool) *IteratorBuilder[T] {
    ib.filters = append(ib.filters, predicate)
    return ib
}

func (ib *IteratorBuilder[T]) Transform(transformer func(T) T) *IteratorBuilder[T] {
    ib.transforms = append(ib.transforms, transformer)
    return ib
}

func (ib *IteratorBuilder[T]) Batch(size int) *IteratorBuilder[T] {
    ib.batchSize = size
    return ib
}

func (ib *IteratorBuilder[T]) Limit(count int) *IteratorBuilder[T] {
    ib.limit = count
    return ib
}

func (ib *IteratorBuilder[T]) Build() Iterator[T] {
    iterator := ib.baseIterator

    // Apply filters
    for _, filter := range ib.filters {
        iterator = NewFilteredIterator(iterator, filter)
    }

    // Apply transforms
    for _, transform := range ib.transforms {
        iterator = NewTransformedIterator(iterator, transform)
    }

    // Apply limit
    if ib.limit > 0 {
        iterator = NewLimitedIterator(iterator, ib.limit)
    }

    return iterator
}
```

### 2. Strategy Pattern Integration

```go
type IterationStrategy[T any] interface {
    CreateIterator(collection Collection[T]) Iterator[T]
}

type ForwardIterationStrategy[T any] struct{}

func (f *ForwardIterationStrategy[T]) CreateIterator(collection Collection[T]) Iterator[T] {
    return collection.CreateIterator()
}

type ReverseIterationStrategy[T any] struct{}

func (r *ReverseIterationStrategy[T]) CreateIterator(collection Collection[T]) Iterator[T] {
    return NewReverseIterator(collection.CreateIterator())
}

type RandomIterationStrategy[T any] struct {
    seed int64
}

func (r *RandomIterationStrategy[T]) CreateIterator(collection Collection[T]) Iterator[T] {
    return NewRandomIterator(collection.CreateIterator(), r.seed)
}

type IterationContext[T any] struct {
    collection Collection[T]
    strategy   IterationStrategy[T]
}

func (ic *IterationContext[T]) SetStrategy(strategy IterationStrategy[T]) {
    ic.strategy = strategy
}

func (ic *IterationContext[T]) Iterate() Iterator[T] {
    return ic.strategy.CreateIterator(ic.collection)
}
```

### 3. Observer Pattern Integration

```go
type IteratorObserver[T any] interface {
    OnNext(item T)
    OnError(err error)
    OnComplete()
}

type ObservableIterator[T any] struct {
    baseIterator Iterator[T]
    observers    []IteratorObserver[T]
}

func NewObservableIterator[T any](baseIterator Iterator[T]/) *ObservableIterator[T] {
    return &ObservableIterator[T]{
        baseIterator: baseIterator,
        observers:    make([]IteratorObserver[T], 0),
    }
}

func (o *ObservableIterator[T]) Subscribe(observer IteratorObserver[T]) {
    o.observers = append(o.observers, observer)
}

func (o *ObservableIterator[T]) Next() (T, error) {
    item, err := o.baseIterator.Next()

    if err != nil {
        for _, observer := range o.observers {
            observer.OnError(err)
        }
        return item, err
    }

    for _, observer := range o.observers {
        observer.OnNext(item)
    }

    if !o.baseIterator.HasNext() {
        for _, observer := range o.observers {
            observer.OnComplete()
        }
    }

    return item, nil
}
```

### 4. Command Pattern Integration

```go
type IteratorCommand interface {
    Execute() error
    Undo() error
}

type ForEachCommand[T any] struct {
    iterator Iterator[T]
    action   func(T) error
    executed []T
}

func NewForEachCommand[T any](iterator Iterator[T], action func(T/) error) *ForEachCommand[T] {
    return &ForEachCommand[T]{
        iterator: iterator,
        action:   action,
        executed: make([]T, 0),
    }
}

func (f *ForEachCommand[T]) Execute() error {
    for f.iterator.HasNext() {
        item, err := f.iterator.Next()
        if err != nil {
            return err
        }

        if err := f.action(item); err != nil {
            return err
        }

        f.executed = append(f.executed, item)
    }

    return nil
}

func (f *ForEachCommand[T]) Undo() error {
    // Implement undo logic if possible
    return fmt.Errorf("undo not supported")
}
```

## Common Interview Questions

### 1. **How does Iterator pattern differ from simple array indexing?**

**Answer:**
Iterator pattern provides abstraction and flexibility that simple indexing cannot:

**Simple Array Indexing:**

```go
// Direct indexing - tightly coupled to array
func ProcessArray(arr []string) {
    for i := 0; i < len(arr); i++ {
        item := arr[i]
        process(item)
    }
}

// Problems:
// 1. Only works with arrays/slices
// 2. Exposes array structure
// 3. No support for lazy loading
// 4. Cannot easily filter or transform
// 5. No support for different traversal orders
```

**Iterator Pattern:**

```go
// Iterator - works with any collection
func ProcessCollection(iterator Iterator[string]) {
    for iterator.HasNext() {
        item, err := iterator.Next()
        if err != nil {
            break
        }
        process(item)
    }
}

// Benefits:
// 1. Works with any collection type (list, tree, database results)
// 2. Hides implementation details
// 3. Supports lazy loading and streaming
// 4. Easy to add filtering and transformation
// 5. Support multiple traversal algorithms
// 6. Memory efficient for large datasets
```

**Key Differences:**

| Aspect                | Array Indexing                  | Iterator Pattern                  |
| --------------------- | ------------------------------- | --------------------------------- |
| **Abstraction**       | Low - exposes array structure   | High - hides collection details   |
| **Flexibility**       | Limited to indexed collections  | Works with any collection         |
| **Memory Usage**      | Entire array in memory          | Can stream/lazy load              |
| **Performance**       | Direct access - fastest         | Slight overhead for abstraction   |
| **Extensibility**     | Hard to extend                  | Easy to add filters/transforms    |
| **Concurrent Access** | Requires manual synchronization | Can provide thread-safe iterators |

### 2. **How do you handle concurrent modification during iteration?**

**Answer:**
Concurrent modification is a common problem that can be handled in several ways:

**Fail-Fast Iterator:**

```go
type FailFastIterator[T any] struct {
    collection      *ObservableCollection[T]
    expectedVersion int64
    currentIndex    int
}

type ObservableCollection[T any] struct {
    items     []T
    version   int64
    observers []CollectionObserver[T]
    mu        sync.RWMutex
}

func (f *FailFastIterator[T]) Next() (T, error) {
    var zero T

    f.collection.mu.RLock()
    currentVersion := f.collection.version
    f.collection.mu.RUnlock()

    if currentVersion != f.expectedVersion {
        return zero, fmt.Errorf("concurrent modification detected")
    }

    // Continue with iteration...
    return f.collection.items[f.currentIndex], nil
}

func (oc *ObservableCollection[T]) Add(item T) {
    oc.mu.Lock()
    defer oc.mu.Unlock()

    oc.items = append(oc.items, item)
    oc.version++

    // Notify observers of modification
    for _, observer := range oc.observers {
        observer.OnItemAdded(item)
    }
}
```

**Copy-on-Write Iterator:**

```go
type CopyOnWriteIterator[T any] struct {
    snapshot []T
    index    int
}

type CopyOnWriteCollection[T any] struct {
    items []T
    mu    sync.RWMutex
}

func (c *CopyOnWriteCollection[T]) CreateIterator() Iterator[T] {
    c.mu.RLock()
    defer c.mu.RUnlock()

    // Create a snapshot for iteration
    snapshot := make([]T, len(c.items))
    copy(snapshot, c.items)

    return &CopyOnWriteIterator[T]{
        snapshot: snapshot,
        index:    0,
    }
}

func (c *CopyOnWriteIterator[T]) Next() (T, error) {
    if c.index >= len(c.snapshot) {
        var zero T
        return zero, fmt.Errorf("no more items")
    }

    item := c.snapshot[c.index]
    c.index++
    return item, nil
}
```

**Synchronized Iterator:**

```go
type SynchronizedIterator[T any] struct {
    collection Collection[T]
    mu         *sync.RWMutex
    index      int
}

func NewSynchronizedIterator[T any](collection Collection[T], mu *sync.RWMutex/) *SynchronizedIterator[T] {
    return &SynchronizedIterator[T]{
        collection: collection,
        mu:         mu,
        index:      0,
    }
}

func (s *SynchronizedIterator[T]) Next() (T, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()

    // Access collection under lock
    if s.index >= s.collection.Size() {
        var zero T
        return zero, fmt.Errorf("no more items")
    }

    // Get item at current index
    item, err := s.collection.Get(s.index)
    s.index++

    return item, err
}
```

**Channel-based Iterator (Go-specific):**

```go
type ChannelIterator[T any] struct {
    channel <-chan T
    current *T
    done    bool
}

func NewChannelIterator[T any](channel <-chan T/) *ChannelIterator[T] {
    return &ChannelIterator[T]{
        channel: channel,
    }
}

func (c *ChannelIterator[T]) HasNext() bool {
    if c.done {
        return false
    }

    if c.current != nil {
        return true
    }

    select {
    case item, ok := <-c.channel:
        if !ok {
            c.done = true
            return false
        }
        c.current = &item
        return true
    default:
        return false
    }
}

func (c *ChannelIterator[T]) Next() (T, error) {
    var zero T

    if !c.HasNext() {
        return zero, fmt.Errorf("no more items")
    }

    item := *c.current
    c.current = nil
    return item, nil
}

// Producer can safely add items via channel
func ProduceItems[T any](channel chan<- T, items []T/) {
    defer close(channel)

    for _, item := range items {
        channel <- item
    }
}
```

### 3. **How do you implement lazy evaluation with iterators?**

**Answer:**
Lazy evaluation allows computation to be deferred until actually needed:

**Lazy Generator Iterator:**

```go
type LazyGeneratorIterator[T any] struct {
    generator func() (T, bool, error)
    cached    *T
    finished  bool
    err       error
}

func NewLazyGeneratorIterator[T any](generator func(/) (T, bool, error)) *LazyGeneratorIterator[T] {
    return &LazyGeneratorIterator[T]{
        generator: generator,
    }
}

func (l *LazyGeneratorIterator[T]) HasNext() bool {
    if l.finished {
        return false
    }

    if l.cached != nil {
        return true
    }

    item, hasNext, err := l.generator()
    if err != nil {
        l.err = err
        l.finished = true
        return false
    }

    if !hasNext {
        l.finished = true
        return false
    }

    l.cached = &item
    return true
}

func (l *LazyGeneratorIterator[T]) Next() (T, error) {
    var zero T

    if l.err != nil {
        return zero, l.err
    }

    if !l.HasNext() {
        return zero, fmt.Errorf("no more items")
    }

    item := *l.cached
    l.cached = nil
    return item, nil
}

// Example: Lazy Fibonacci sequence
func CreateLazyFibonacci() Iterator[int] {
    a, b := 0, 1

    generator := func() (int, bool, error) {
        current := a
        a, b = b, a+b
        return current, true, nil // Infinite sequence
    }

    return NewLazyGeneratorIterator(generator)
}
```

**Lazy Transformation Iterator:**

```go
type LazyTransformIterator[T, U any] struct {
    sourceIterator Iterator[T]
    transform      func(T) U
    filter         func(T) bool
    computed       *U
    sourceItem     *T
}

func NewLazyTransformIterator[T, U any](README.md) U,
    filter func(T) bool,
) *LazyTransformIterator[T, U] {
    return &LazyTransformIterator[T, U]{
        sourceIterator: source,
        transform:      transform,
        filter:         filter,
    }
}

func (l *LazyTransformIterator[T, U]) HasNext() bool {
    if l.computed != nil {
        return true
    }

    return l.findNextValidItem()
}

func (l *LazyTransformIterator[T, U]) Next() (U, error) {
    var zero U

    if !l.HasNext() {
        return zero, fmt.Errorf("no more items")
    }

    result := *l.computed
    l.computed = nil
    l.sourceItem = nil

    return result, nil
}

func (l *LazyTransformIterator[T, U]) findNextValidItem() bool {
    for l.sourceIterator.HasNext() {
        item, err := l.sourceIterator.Next()
        if err != nil {
            return false
        }

        // Apply filter
        if l.filter != nil && !l.filter(item) {
            continue
        }

        // Apply transformation (lazy - only when needed)
        transformed := l.transform(item)
        l.computed = &transformed
        l.sourceItem = &item

        return true
    }

    return false
}
```

**Database Result Set Iterator (Lazy Loading):**

```go
type LazyDatabaseIterator[T any] struct {
    db           Database
    query        string
    args         []interface{}
    batchSize    int
    currentBatch []T
    batchIndex   int
    offset       int
    hasMore      bool
    mapper       func(*sql.Rows) (T, error)
}

func NewLazyDatabaseIterator[T any](README.md) (T, error),
) *LazyDatabaseIterator[T] {
    return &LazyDatabaseIterator[T]{
        db:        db,
        query:     query,
        args:      args,
        batchSize: batchSize,
        hasMore:   true,
        mapper:    mapper,
    }
}

func (l *LazyDatabaseIterator[T]) HasNext() bool {
    if l.batchIndex < len(l.currentBatch) {
        return true
    }

    if !l.hasMore {
        return false
    }

    // Lazy load next batch only when needed
    return l.loadNextBatch() == nil
}

func (l *LazyDatabaseIterator[T]) Next() (T, error) {
    var zero T

    if !l.HasNext() {
        return zero, fmt.Errorf("no more items")
    }

    item := l.currentBatch[l.batchIndex]
    l.batchIndex++

    return item, nil
}

func (l *LazyDatabaseIterator[T]) loadNextBatch() error {
    // Add LIMIT and OFFSET to query
    paginatedQuery := l.query + " LIMIT ? OFFSET ?"
    queryArgs := append(l.args, l.batchSize, l.offset)

    rows, err := l.db.Query(paginatedQuery, queryArgs...)
    if err != nil {
        return err
    }
    defer rows.Close()

    var batch []T
    for rows.Next() {
        item, err := l.mapper(rows)
        if err != nil {
            return err
        }
        batch = append(batch, item)
    }

    l.currentBatch = batch
    l.batchIndex = 0
    l.offset += len(batch)
    l.hasMore = len(batch) == l.batchSize

    return nil
}
```

### 4. **How do you implement iterator composition and chaining?**

**Answer:**
Iterator composition allows building complex iteration logic from simple components:

**Fluent Iterator API:**

```go
type FluentIterator[T any] struct {
    baseIterator Iterator[T]
}

func Fluent[T any](iterator Iterator[T]/) *FluentIterator[T] {
    return &FluentIterator[T]{baseIterator: iterator}
}

func (f *FluentIterator[T]) Filter(predicate func(T) bool) *FluentIterator[T] {
    return &FluentIterator[T]{
        baseIterator: NewFilteredIterator(f.baseIterator, predicate),
    }
}

func (f *FluentIterator[T]) Take(count int) *FluentIterator[T] {
    return &FluentIterator[T]{
        baseIterator: NewLimitedIterator(f.baseIterator, count),
    }
}

func (f *FluentIterator[T]) Skip(count int) *FluentIterator[T] {
    return &FluentIterator[T]{
        baseIterator: NewSkipIterator(f.baseIterator, count),
    }
}

func (f *FluentIterator[T]) Map(mapper func(T) T) *FluentIterator[T] {
    return &FluentIterator[T]{
        baseIterator: NewTransformedIterator(f.baseIterator, mapper),
    }
}

func (f *FluentIterator[T]) ToSlice() []T {
    var result []T
    for f.baseIterator.HasNext() {
        item, err := f.baseIterator.Next()
        if err != nil {
            break
        }
        result = append(result, item)
    }
    return result
}

func (f *FluentIterator[T]) Reduce(initial T, reducer func(T, T) T) T {
    accumulator := initial
    for f.baseIterator.HasNext() {
        item, err := f.baseIterator.Next()
        if err != nil {
            break
        }
        accumulator = reducer(accumulator, item)
    }
    return accumulator
}

// Usage example:
// result := Fluent(collection.CreateIterator()).
//     Filter(func(x int) bool { return x > 0 }).
//     Map(func(x int) int { return x * 2 }).
//     Take(10).
//     ToSlice()
```

**Pipeline Iterator:**

```go
type PipelineStage[T any] interface {
    Process(input Iterator[T]) Iterator[T]
}

type FilterStage[T any] struct {
    predicate func(T) bool
}

func (f *FilterStage[T]) Process(input Iterator[T]) Iterator[T] {
    return NewFilteredIterator(input, f.predicate)
}

type MapStage[T any] struct {
    mapper func(T) T
}

func (m *MapStage[T]) Process(input Iterator[T]) Iterator[T] {
    return NewTransformedIterator(input, m.mapper)
}

type IteratorPipeline[T any] struct {
    stages []PipelineStage[T]
}

func NewIteratorPipeline[T any]() *IteratorPipeline[T] {
    return &IteratorPipeline[T]{
        stages: make([]PipelineStage[T], 0),
    }
}

func (ip *IteratorPipeline[T]) AddStage(stage PipelineStage[T]) *IteratorPipeline[T] {
    ip.stages = append(ip.stages, stage)
    return ip
}

func (ip *IteratorPipeline[T]) Execute(input Iterator[T]) Iterator[T] {
    current := input

    for _, stage := range ip.stages {
        current = stage.Process(current)
    }

    return current
}

// Usage:
// pipeline := NewIteratorPipeline[int]().
//     AddStage(&FilterStage[int]{predicate: isPositive}).
//     AddStage(&MapStage[int]{mapper: double})
//
// result := pipeline.Execute(collection.CreateIterator())
```

**Parallel Iterator Composition:**

```go
type ParallelIterator[T any] struct {
    iterators []Iterator[T]
    channels  []<-chan IteratorItem[T]
    done      chan struct{}
    merger    chan IteratorItem[T]
}

type IteratorItem[T any] struct {
    Value T
    Error error
    Done  bool
}

func NewParallelIterator[T any](iterators ...Iterator[T]/) *ParallelIterator[T] {
    pi := &ParallelIterator[T]{
        iterators: iterators,
        channels:  make([]<-chan IteratorItem[T], len(iterators)),
        done:      make(chan struct{}),
        merger:    make(chan IteratorItem[T], len(iterators)),
    }

    // Start goroutine for each iterator
    for i, iterator := range iterators {
        channel := make(chan IteratorItem[T])
        pi.channels[i] = channel

        go func(iter Iterator[T], ch chan<- IteratorItem[T]) {
            defer close(ch)

            for iter.HasNext() {
                item, err := iter.Next()
                select {
                case ch <- IteratorItem[T]{Value: item, Error: err}:
                case <-pi.done:
                    return
                }

                if err != nil {
                    return
                }
            }

            ch <- IteratorItem[T]{Done: true}
        }(iterator, channel)
    }

    // Start merger goroutine
    go pi.merge()

    return pi
}

func (pi *ParallelIterator[T]) merge() {
    defer close(pi.merger)

    activeChannels := len(pi.channels)

    for activeChannels > 0 {
        for i, ch := range pi.channels {
            if ch == nil {
                continue
            }

            select {
            case item, ok := <-ch:
                if !ok || item.Done {
                    pi.channels[i] = nil
                    activeChannels--
                } else {
                    pi.merger <- item
                }
            case <-pi.done:
                return
            default:
                // Non-blocking, continue to next channel
            }
        }
    }
}

func (pi *ParallelIterator[T]) HasNext() bool {
    select {
    case _, ok := <-pi.merger:
        return ok
    default:
        return true
    }
}

func (pi *ParallelIterator[T]) Next() (T, error) {
    var zero T

    select {
    case item, ok := <-pi.merger:
        if !ok {
            return zero, fmt.Errorf("no more items")
        }
        return item.Value, item.Error
    case <-pi.done:
        return zero, fmt.Errorf("iterator closed")
    }
}

func (pi *ParallelIterator[T]) Close() error {
    close(pi.done)
    return nil
}
```

### 5. **When should you avoid using Iterator pattern?**

**Answer:**
Iterator pattern should be avoided in certain scenarios:

**Simple Data Structures:**

```go
// DON'T use iterator for simple arrays when direct access is sufficient
type SimpleProcessor struct{}

func (s *SimpleProcessor) ProcessNumbers(numbers []int) int {
    sum := 0
    for _, num := range numbers { // Direct range iteration is simpler
        sum += num
    }
    return sum
}

// Instead of unnecessary iterator:
// iterator := NewSliceIterator(numbers)
// for iterator.HasNext() { ... } // Overkill for simple case
```

**Performance-Critical Code:**

```go
// DON'T use iterator in performance-critical loops
func ProcessLargeArray(data []float64) float64 {
    sum := 0.0

    // Direct array access is faster
    for i := 0; i < len(data); i++ {
        sum += data[i] * data[i] // Hot path - direct access
    }

    return sum

    // Iterator would add unnecessary overhead:
    // iterator := NewSliceIterator(data)
    // for iterator.HasNext() { ... } // Slower due to method calls
}
```

**Small, Fixed Collections:**

```go
// DON'T use iterator for small, known collections
type Colors struct {
    Primary []string
}

func (c *Colors) GetPrimaryColors() []string {
    // Direct return is simpler than iterator
    return []string{"Red", "Blue", "Yellow"}
}

// No need for:
// func (c *Colors) CreateIterator() Iterator[string] { ... }
```

**Single-Use, Simple Iterations:**

```go
// DON'T use iterator for one-off, simple operations
func FindMax(numbers []int) int {
    if len(numbers) == 0 {
        return 0
    }

    max := numbers[0]
    for _, num := range numbers { // Simple range is clearer
        if num > max {
            max = num
        }
    }

    return max
}
```

**Better Alternatives:**

| Scenario              | Alternative                                       | Reason                    |
| --------------------- | ------------------------------------------------- | ------------------------- |
| Simple arrays         | Range loops (`for _, v := range slice`)           | Less overhead             |
| Performance critical  | Direct indexing (`for i := 0; i < len(arr); i++`) | Fastest access            |
| Functional operations | Higher-order functions (map, filter, reduce)      | More expressive           |
| Stream processing     | Channels and goroutines                           | Better for Go concurrency |
| One-time operations   | Direct loops                                      | Simpler code              |
| Small collections     | Direct methods                                    | Less abstraction overhead |

**Decision Framework:**

```go
type IteratorDecision struct {
    CollectionSize      int
    AccessPattern       string // "sequential", "random", "filtered"
    PerformanceNeeds    string // "low", "medium", "high"
    ReuseRequirements   bool
    LazyLoadingNeeds    bool
    MultipleTraversals  bool
    CollectionComplexity string // "simple", "complex"
}

func (id *IteratorDecision) ShouldUseIterator() (bool, string) {
    if id.CollectionSize < 10 && id.CollectionComplexity == "simple" {
        return false, "Collection too simple for iterator pattern"
    }

    if id.PerformanceNeeds == "high" && id.AccessPattern == "sequential" {
        return false, "Direct iteration faster for performance-critical sequential access"
    }

    if !id.ReuseRequirements && !id.LazyLoadingNeeds && !id.MultipleTraversals {
        return false, "No clear benefits over direct iteration"
    }

    if id.LazyLoadingNeeds || id.MultipleTraversals || id.AccessPattern == "filtered" {
        return true, "Iterator provides valuable abstraction and functionality"
    }

    if id.CollectionComplexity == "complex" {
        return true, "Iterator hides complex collection structure"
    }

    return false, "Direct iteration likely sufficient"
}
```
