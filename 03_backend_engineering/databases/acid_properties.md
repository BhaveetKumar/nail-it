# ACID Properties - Database Transaction Guarantees

## Overview

ACID properties are fundamental guarantees that database transactions must satisfy to ensure data integrity and reliability. These properties are crucial for maintaining data consistency in financial systems like payment processing.

## Key Concepts

- **Atomicity**: All or nothing - transactions are indivisible
- **Consistency**: Database remains in valid state before and after transaction
- **Isolation**: Concurrent transactions don't interfere with each other
- **Durability**: Committed changes persist even after system failures

## ACID Properties Deep Dive

### 1. Atomicity
- **Definition**: Transaction is treated as a single unit of work
- **Implementation**: Rollback mechanism for failed operations
- **Example**: Money transfer - both debit and credit must succeed or both fail

### 2. Consistency
- **Definition**: Database constraints and rules are maintained
- **Implementation**: Validation checks and constraint enforcement
- **Example**: Account balance cannot go negative

### 3. Isolation
- **Definition**: Concurrent transactions don't see each other's uncommitted changes
- **Implementation**: Locking mechanisms and isolation levels
- **Example**: Two users transferring money simultaneously

### 4. Durability
- **Definition**: Committed changes survive system failures
- **Implementation**: Write-ahead logging and persistent storage
- **Example**: Payment confirmation survives power outage

## Go Implementation

```go
package main

import (
    "context"
    "database/sql"
    "fmt"
    "log"
    "time"

    _ "github.com/lib/pq"
)

// TransactionManager manages database transactions
type TransactionManager struct {
    db *sql.DB
}

// NewTransactionManager creates a new transaction manager
func NewTransactionManager(db *sql.DB) *TransactionManager {
    return &TransactionManager{db: db}
}

// Account represents a bank account
type Account struct {
    ID      int     `json:"id"`
    Balance float64 `json:"balance"`
    Owner   string  `json:"owner"`
}

// TransferRequest represents a money transfer request
type TransferRequest struct {
    FromAccountID int     `json:"from_account_id"`
    ToAccountID   int     `json:"to_account_id"`
    Amount        float64 `json:"amount"`
    Description   string  `json:"description"`
}

// TransferResult represents the result of a transfer
type TransferResult struct {
    Success      bool      `json:"success"`
    TransactionID string   `json:"transaction_id"`
    Error        string    `json:"error,omitempty"`
    Timestamp    time.Time `json:"timestamp"`
}

// TransferMoney performs a money transfer with ACID properties
func (tm *TransactionManager) TransferMoney(ctx context.Context, req TransferRequest) (*TransferResult, error) {
    // Start transaction
    tx, err := tm.db.BeginTx(ctx, &sql.TxOptions{
        Isolation: sql.LevelSerializable, // Highest isolation level
    })
    if err != nil {
        return nil, fmt.Errorf("failed to begin transaction: %v", err)
    }
    defer tx.Rollback() // Ensure rollback if not committed

    // Generate transaction ID
    transactionID := fmt.Sprintf("txn_%d", time.Now().UnixNano())

    // ATOMICITY: All operations must succeed or all must fail
    if err := tm.validateTransfer(ctx, tx, req); err != nil {
        return &TransferResult{
            Success:      false,
            TransactionID: transactionID,
            Error:        err.Error(),
            Timestamp:    time.Now(),
        }, nil
    }

    // Debit from source account
    if err := tm.debitAccount(ctx, tx, req.FromAccountID, req.Amount, transactionID); err != nil {
        return &TransferResult{
            Success:      false,
            TransactionID: transactionID,
            Error:        fmt.Sprintf("debit failed: %v", err),
            Timestamp:    time.Now(),
        }, nil
    }

    // Credit to destination account
    if err := tm.creditAccount(ctx, tx, req.ToAccountID, req.Amount, transactionID); err != nil {
        return &TransferResult{
            Success:      false,
            TransactionID: transactionID,
            Error:        fmt.Sprintf("credit failed: %v", err),
            Timestamp:    time.Now(),
        }, nil
    }

    // Log transaction
    if err := tm.logTransaction(ctx, tx, transactionID, req); err != nil {
        return &TransferResult{
            Success:      false,
            TransactionID: transactionID,
            Error:        fmt.Sprintf("logging failed: %v", err),
            Timestamp:    time.Now(),
        }, nil
    }

    // COMMIT: Make all changes permanent
    if err := tx.Commit(); err != nil {
        return &TransferResult{
            Success:      false,
            TransactionID: transactionID,
            Error:        fmt.Sprintf("commit failed: %v", err),
            Timestamp:    time.Now(),
        }, nil
    }

    return &TransferResult{
        Success:      true,
        TransactionID: transactionID,
        Timestamp:    time.Now(),
    }, nil
}

// validateTransfer validates the transfer request
func (tm *TransactionManager) validateTransfer(ctx context.Context, tx *sql.Tx, req TransferRequest) error {
    // CONSISTENCY: Check business rules and constraints
    
    // Check if amount is positive
    if req.Amount <= 0 {
        return fmt.Errorf("transfer amount must be positive")
    }

    // Check if accounts exist and are different
    if req.FromAccountID == req.ToAccountID {
        return fmt.Errorf("cannot transfer to the same account")
    }

    // Check source account exists and has sufficient balance
    var balance float64
    err := tx.QueryRowContext(ctx, 
        "SELECT balance FROM accounts WHERE id = $1", 
        req.FromAccountID).Scan(&balance)
    if err != nil {
        if err == sql.ErrNoRows {
            return fmt.Errorf("source account not found")
        }
        return fmt.Errorf("failed to check source account: %v", err)
    }

    if balance < req.Amount {
        return fmt.Errorf("insufficient balance: required %.2f, available %.2f", 
            req.Amount, balance)
    }

    // Check destination account exists
    var destAccountID int
    err = tx.QueryRowContext(ctx, 
        "SELECT id FROM accounts WHERE id = $1", 
        req.ToAccountID).Scan(&destAccountID)
    if err != nil {
        if err == sql.ErrNoRows {
            return fmt.Errorf("destination account not found")
        }
        return fmt.Errorf("failed to check destination account: %v", err)
    }

    return nil
}

// debitAccount debits money from an account
func (tm *TransactionManager) debitAccount(ctx context.Context, tx *sql.Tx, accountID int, amount float64, transactionID string) error {
    // ISOLATION: This operation is isolated from other concurrent transactions
    
    result, err := tx.ExecContext(ctx, 
        "UPDATE accounts SET balance = balance - $1 WHERE id = $2 AND balance >= $1",
        amount, accountID)
    if err != nil {
        return fmt.Errorf("failed to debit account: %v", err)
    }

    rowsAffected, err := result.RowsAffected()
    if err != nil {
        return fmt.Errorf("failed to get rows affected: %v", err)
    }

    if rowsAffected == 0 {
        return fmt.Errorf("account not found or insufficient balance")
    }

    // Log the debit operation
    _, err = tx.ExecContext(ctx, 
        "INSERT INTO transaction_logs (transaction_id, account_id, amount, type, timestamp) VALUES ($1, $2, $3, $4, $5)",
        transactionID, accountID, -amount, "debit", time.Now())
    if err != nil {
        return fmt.Errorf("failed to log debit: %v", err)
    }

    return nil
}

// creditAccount credits money to an account
func (tm *TransactionManager) creditAccount(ctx context.Context, tx *sql.Tx, accountID int, amount float64, transactionID string) error {
    // ISOLATION: This operation is isolated from other concurrent transactions
    
    result, err := tx.ExecContext(ctx, 
        "UPDATE accounts SET balance = balance + $1 WHERE id = $2",
        amount, accountID)
    if err != nil {
        return fmt.Errorf("failed to credit account: %v", err)
    }

    rowsAffected, err := result.RowsAffected()
    if err != nil {
        return fmt.Errorf("failed to get rows affected: %v", err)
    }

    if rowsAffected == 0 {
        return fmt.Errorf("destination account not found")
    }

    // Log the credit operation
    _, err = tx.ExecContext(ctx, 
        "INSERT INTO transaction_logs (transaction_id, account_id, amount, type, timestamp) VALUES ($1, $2, $3, $4, $5)",
        transactionID, accountID, amount, "credit", time.Now())
    if err != nil {
        return fmt.Errorf("failed to log credit: %v", err)
    }

    return nil
}

// logTransaction logs the complete transaction
func (tm *TransactionManager) logTransaction(ctx context.Context, tx *sql.Tx, transactionID string, req TransferRequest) error {
    // DURABILITY: Transaction details are logged for audit and recovery
    
    _, err := tx.ExecContext(ctx, 
        `INSERT INTO transactions (id, from_account_id, to_account_id, amount, description, status, created_at) 
         VALUES ($1, $2, $3, $4, $5, $6, $7)`,
        transactionID, req.FromAccountID, req.ToAccountID, req.Amount, req.Description, "completed", time.Now())
    if err != nil {
        return fmt.Errorf("failed to log transaction: %v", err)
    }

    return nil
}

// GetAccountBalance gets the current balance of an account
func (tm *TransactionManager) GetAccountBalance(ctx context.Context, accountID int) (float64, error) {
    var balance float64
    err := tm.db.QueryRowContext(ctx, 
        "SELECT balance FROM accounts WHERE id = $1", 
        accountID).Scan(&balance)
    if err != nil {
        if err == sql.ErrNoRows {
            return 0, fmt.Errorf("account not found")
        }
        return 0, fmt.Errorf("failed to get account balance: %v", err)
    }
    return balance, nil
}

// GetTransactionHistory gets transaction history for an account
func (tm *TransactionManager) GetTransactionHistory(ctx context.Context, accountID int, limit int) ([]map[string]interface{}, error) {
    query := `
        SELECT transaction_id, account_id, amount, type, timestamp
        FROM transaction_logs 
        WHERE account_id = $1 
        ORDER BY timestamp DESC 
        LIMIT $2
    `
    
    rows, err := tm.db.QueryContext(ctx, query, accountID, limit)
    if err != nil {
        return nil, fmt.Errorf("failed to get transaction history: %v", err)
    }
    defer rows.Close()

    var transactions []map[string]interface{}
    for rows.Next() {
        var transactionID string
        var accID int
        var amount float64
        var transactionType string
        var timestamp time.Time

        if err := rows.Scan(&transactionID, &accID, &amount, &transactionType, &timestamp); err != nil {
            return nil, fmt.Errorf("failed to scan transaction: %v", err)
        }

        transaction := map[string]interface{}{
            "transaction_id": transactionID,
            "account_id":     accID,
            "amount":         amount,
            "type":           transactionType,
            "timestamp":      timestamp,
        }
        transactions = append(transactions, transaction)
    }

    return transactions, nil
}

// Example usage
func main() {
    // Database connection (replace with your actual connection string)
    db, err := sql.Open("postgres", "user=postgres password=password dbname=banking sslmode=disable")
    if err != nil {
        log.Fatalf("Failed to connect to database: %v", err)
    }
    defer db.Close()

    // Test connection
    if err := db.Ping(); err != nil {
        log.Fatalf("Failed to ping database: %v", err)
    }

    // Create transaction manager
    tm := NewTransactionManager(db)

    // Create sample accounts
    ctx := context.Background()
    
    // Insert sample accounts
    _, err = db.ExecContext(ctx, `
        INSERT INTO accounts (id, balance, owner) VALUES 
        (1, 1000.00, 'Alice'),
        (2, 500.00, 'Bob')
        ON CONFLICT (id) DO UPDATE SET balance = EXCLUDED.balance
    `)
    if err != nil {
        log.Printf("Failed to insert sample accounts: %v", err)
    }

    // Perform a transfer
    transferReq := TransferRequest{
        FromAccountID: 1,
        ToAccountID:   2,
        Amount:        100.00,
        Description:   "Payment for services",
    }

    result, err := tm.TransferMoney(ctx, transferReq)
    if err != nil {
        log.Printf("Transfer failed: %v", err)
    } else {
        fmt.Printf("Transfer result: %+v\n", result)
    }

    // Check account balances
    balance1, err := tm.GetAccountBalance(ctx, 1)
    if err != nil {
        log.Printf("Failed to get balance for account 1: %v", err)
    } else {
        fmt.Printf("Account 1 balance: %.2f\n", balance1)
    }

    balance2, err := tm.GetAccountBalance(ctx, 2)
    if err != nil {
        log.Printf("Failed to get balance for account 2: %v", err)
    } else {
        fmt.Printf("Account 2 balance: %.2f\n", balance2)
    }

    // Get transaction history
    history, err := tm.GetTransactionHistory(ctx, 1, 10)
    if err != nil {
        log.Printf("Failed to get transaction history: %v", err)
    } else {
        fmt.Printf("Transaction history for account 1:\n")
        for _, txn := range history {
            fmt.Printf("  %s: %.2f (%s) at %s\n", 
                txn["transaction_id"], txn["amount"], txn["type"], txn["timestamp"])
        }
    }
}
```

## Isolation Levels

### 1. Read Uncommitted
- **Description**: Can read uncommitted changes
- **Use Case**: Performance critical, data accuracy not critical
- **Problems**: Dirty reads, non-repeatable reads, phantom reads

### 2. Read Committed
- **Description**: Can only read committed changes
- **Use Case**: Default for most databases
- **Problems**: Non-repeatable reads, phantom reads

### 3. Repeatable Read
- **Description**: Same query returns same results within transaction
- **Use Case**: When consistency is important
- **Problems**: Phantom reads

### 4. Serializable
- **Description**: Highest isolation level
- **Use Case**: Financial transactions, critical data
- **Problems**: Performance impact, potential deadlocks

## Benefits

1. **Data Integrity**: Ensures data remains consistent
2. **Reliability**: Transactions are reliable and predictable
3. **Concurrency**: Safe concurrent access to data
4. **Recovery**: System can recover from failures
5. **Audit Trail**: Complete transaction history

## Trade-offs

1. **Performance**: ACID properties can impact performance
2. **Complexity**: Implementation is complex
3. **Scalability**: Can limit scalability in distributed systems
4. **Resource Usage**: Requires more resources
5. **Locking**: Can cause contention and deadlocks

## Use Cases

- **Financial Systems**: Banking, payment processing
- **E-commerce**: Order processing, inventory management
- **Healthcare**: Patient records, medical billing
- **Government**: Tax records, citizen data
- **Critical Systems**: Air traffic control, nuclear systems

## Best Practices

1. **Keep Transactions Short**: Minimize lock time
2. **Use Appropriate Isolation Level**: Balance consistency and performance
3. **Handle Deadlocks**: Implement retry logic
4. **Monitor Performance**: Track transaction metrics
5. **Test Thoroughly**: Verify ACID properties

## Common Pitfalls

1. **Long Transactions**: Can cause deadlocks and performance issues
2. **Wrong Isolation Level**: Too high or too low for use case
3. **Ignoring Errors**: Not handling transaction failures
4. **Nested Transactions**: Can cause unexpected behavior
5. **Resource Leaks**: Not properly closing transactions

## Interview Questions

1. **What are ACID properties?**
   - Atomicity, Consistency, Isolation, Durability - fundamental database transaction guarantees

2. **How do you ensure data consistency in financial transactions?**
   - Use ACID properties, appropriate isolation levels, proper error handling

3. **What is the difference between isolation levels?**
   - Different levels of protection against concurrent access issues

4. **How do you handle deadlocks?**
   - Implement retry logic, use timeouts, optimize transaction order

## Time Complexity

- **Transaction**: O(n) where n is number of operations
- **Locking**: O(1) for single resource
- **Rollback**: O(n) where n is number of operations

## Space Complexity

- **Transaction Log**: O(n) where n is transaction size
- **Lock Storage**: O(m) where m is number of locked resources
- **Rollback Data**: O(n) where n is transaction size

The optimal solution uses:
1. **Proper Isolation**: Choose appropriate isolation level
2. **Short Transactions**: Minimize lock time
3. **Error Handling**: Handle all failure scenarios
4. **Monitoring**: Track transaction performance
