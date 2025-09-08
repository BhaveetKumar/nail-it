# 06. Order Matching Engine - Trading System Core

## Title & Summary
Design and implement an order matching engine for a trading system that handles buy/sell orders, price-time priority matching, and real-time order book management with high performance and low latency.

## Problem Statement

Build a trading order matching engine that:

1. **Order Management**: Handle buy and sell orders with price-time priority
2. **Order Matching**: Match orders based on price and time priority
3. **Order Book**: Maintain real-time order book with bid/ask levels
4. **Trade Execution**: Execute trades and generate trade confirmations
5. **Market Data**: Provide real-time market data and order book updates
6. **Risk Management**: Basic risk checks and order validation

## Requirements & Constraints

### Functional Requirements
- Submit and cancel orders
- Price-time priority matching algorithm
- Real-time order book maintenance
- Trade execution and confirmation
- Market data broadcasting
- Basic risk management

### Non-Functional Requirements
- **Latency**: < 1ms for order matching
- **Consistency**: Strong consistency for order state
- **Memory**: Support 1M active orders
- **Scalability**: Handle 100K orders per second
- **Reliability**: 99.99% order processing success rate

## API / Interfaces

### REST Endpoints

```go
// Order Management
POST   /api/orders/submit
DELETE /api/orders/{orderID}/cancel
GET    /api/orders/{orderID}
GET    /api/orders/user/{userID}

// Market Data
GET    /api/market/book/{symbol}
GET    /api/market/trades/{symbol}
GET    /api/market/price/{symbol}

// WebSocket
WS     /ws/market/{symbol}
WS     /ws/orders/{userID}
```

### Request/Response Examples

```json
// Submit Order
POST /api/orders/submit
{
  "orderID": "order_123",
  "userID": "user_456",
  "symbol": "AAPL",
  "side": "buy",
  "quantity": 100,
  "price": 150.50,
  "orderType": "limit",
  "timeInForce": "GTC"
}

// Order Response
{
  "orderID": "order_123",
  "status": "filled",
  "filledQuantity": 100,
  "remainingQuantity": 0,
  "averagePrice": 150.50,
  "trades": [
    {
      "tradeID": "trade_789",
      "quantity": 100,
      "price": 150.50,
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ]
}
```

## Data Model

### Core Entities

```go
type Order struct {
    ID               string      `json:"id"`
    UserID           string      `json:"userID"`
    Symbol           string      `json:"symbol"`
    Side             OrderSide   `json:"side"`
    Quantity         int64       `json:"quantity"`
    Price            float64     `json:"price"`
    OrderType        OrderType   `json:"orderType"`
    TimeInForce      TimeInForce `json:"timeInForce"`
    Status           OrderStatus `json:"status"`
    FilledQuantity   int64       `json:"filledQuantity"`
    RemainingQuantity int64      `json:"remainingQuantity"`
    AveragePrice     float64     `json:"averagePrice"`
    CreatedAt        time.Time   `json:"createdAt"`
    UpdatedAt        time.Time   `json:"updatedAt"`
}

type Trade struct {
    ID           string    `json:"id"`
    BuyOrderID   string    `json:"buyOrderID"`
    SellOrderID  string    `json:"sellOrderID"`
    Symbol       string    `json:"symbol"`
    Quantity     int64     `json:"quantity"`
    Price        float64   `json:"price"`
    Timestamp    time.Time `json:"timestamp"`
}

type OrderBook struct {
    Symbol    string      `json:"symbol"`
    Bids      []OrderLevel `json:"bids"`
    Asks      []OrderLevel `json:"asks"`
    LastPrice float64     `json:"lastPrice"`
    Volume    int64       `json:"volume"`
    Timestamp time.Time   `json:"timestamp"`
}

type OrderLevel struct {
    Price    float64 `json:"price"`
    Quantity int64   `json:"quantity"`
    Orders   int     `json:"orders"`
}
```

## Approach Overview

### Simple Solution (MVP)
1. In-memory order storage with simple matching
2. Basic price-time priority
3. Simple order book representation
4. No risk management or advanced features

### Production-Ready Design
1. **High-Performance Engine**: Lock-free data structures and algorithms
2. **Order Book Management**: Efficient bid/ask level management
3. **Matching Algorithm**: Price-time priority with partial fills
4. **Risk Management**: Real-time risk checks and position limits
5. **Market Data**: Real-time market data distribution
6. **Persistence**: Order and trade persistence with recovery

## Detailed Design

### Modular Decomposition

```go
ordermatching/
├── engine/        # Core matching engine
├── orderbook/     # Order book management
├── orders/        # Order management
├── trades/        # Trade execution
├── risk/          # Risk management
├── marketdata/    # Market data distribution
└── persistence/   # Data persistence
```

### Concurrency Model

```go
type MatchingEngine struct {
    orderBooks    map[string]*OrderBook
    orders        map[string]*Order
    trades        []Trade
    riskManager   *RiskManager
    marketData    *MarketDataService
    mutex         sync.RWMutex
    orderChan     chan OrderRequest
    cancelChan    chan CancelRequest
    tradeChan     chan Trade
}

// Goroutines for:
// 1. Order processing
// 2. Trade execution
// 3. Market data broadcasting
// 4. Risk management
```

## Optimal Golang Implementation

```go
package main

import (
    "container/heap"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "sort"
    "sync"
    "time"

    "github.com/google/uuid"
)

type OrderSide string
const (
    SideBuy  OrderSide = "buy"
    SideSell OrderSide = "sell"
)

type OrderType string
const (
    TypeLimit  OrderType = "limit"
    TypeMarket OrderType = "market"
)

type TimeInForce string
const (
    TIFGTC TimeInForce = "GTC" // Good Till Cancelled
    TIFIOC TimeInForce = "IOC" // Immediate or Cancel
    TIFFOK TimeInForce = "FOK" // Fill or Kill
)

type OrderStatus string
const (
    StatusPending   OrderStatus = "pending"
    StatusPartiallyFilled OrderStatus = "partially_filled"
    StatusFilled    OrderStatus = "filled"
    StatusCancelled OrderStatus = "cancelled"
    StatusRejected  OrderStatus = "rejected"
)

type Order struct {
    ID               string      `json:"id"`
    UserID           string      `json:"userID"`
    Symbol           string      `json:"symbol"`
    Side             OrderSide   `json:"side"`
    Quantity         int64       `json:"quantity"`
    Price            float64     `json:"price"`
    OrderType        OrderType   `json:"orderType"`
    TimeInForce      TimeInForce `json:"timeInForce"`
    Status           OrderStatus `json:"status"`
    FilledQuantity   int64       `json:"filledQuantity"`
    RemainingQuantity int64      `json:"remainingQuantity"`
    AveragePrice     float64     `json:"averagePrice"`
    CreatedAt        time.Time   `json:"createdAt"`
    UpdatedAt        time.Time   `json:"updatedAt"`
}

type Trade struct {
    ID           string    `json:"id"`
    BuyOrderID   string    `json:"buyOrderID"`
    SellOrderID  string    `json:"sellOrderID"`
    Symbol       string    `json:"symbol"`
    Quantity     int64     `json:"quantity"`
    Price        float64   `json:"price"`
    Timestamp    time.Time `json:"timestamp"`
}

type OrderLevel struct {
    Price    float64 `json:"price"`
    Quantity int64   `json:"quantity"`
    Orders   int     `json:"orders"`
}

type OrderBook struct {
    Symbol    string       `json:"symbol"`
    Bids      []OrderLevel `json:"bids"`
    Asks      []OrderLevel `json:"asks"`
    LastPrice float64      `json:"lastPrice"`
    Volume    int64        `json:"volume"`
    Timestamp time.Time    `json:"timestamp"`
}

// Priority queue for orders
type OrderQueue struct {
    orders []*Order
    mutex  sync.RWMutex
}

func (oq *OrderQueue) Len() int {
    oq.mutex.RLock()
    defer oq.mutex.RUnlock()
    return len(oq.orders)
}

func (oq *OrderQueue) Less(i, j int) bool {
    oq.mutex.RLock()
    defer oq.mutex.RUnlock()
    
    // For buy orders: higher price first, then earlier time
    // For sell orders: lower price first, then earlier time
    if oq.orders[i].Side == SideBuy {
        if oq.orders[i].Price != oq.orders[j].Price {
            return oq.orders[i].Price > oq.orders[j].Price
        }
    } else {
        if oq.orders[i].Price != oq.orders[j].Price {
            return oq.orders[i].Price < oq.orders[j].Price
        }
    }
    
    return oq.orders[i].CreatedAt.Before(oq.orders[j].CreatedAt)
}

func (oq *OrderQueue) Swap(i, j int) {
    oq.mutex.Lock()
    defer oq.mutex.Unlock()
    oq.orders[i], oq.orders[j] = oq.orders[j], oq.orders[i]
}

func (oq *OrderQueue) Push(x interface{}) {
    oq.mutex.Lock()
    defer oq.mutex.Unlock()
    oq.orders = append(oq.orders, x.(*Order))
}

func (oq *OrderQueue) Pop() interface{} {
    oq.mutex.Lock()
    defer oq.mutex.Unlock()
    n := len(oq.orders)
    order := oq.orders[n-1]
    oq.orders = oq.orders[0 : n-1]
    return order
}

func (oq *OrderQueue) Peek() *Order {
    oq.mutex.RLock()
    defer oq.mutex.RUnlock()
    if len(oq.orders) == 0 {
        return nil
    }
    return oq.orders[0]
}

func (oq *OrderQueue) Remove(orderID string) bool {
    oq.mutex.Lock()
    defer oq.mutex.Unlock()
    
    for i, order := range oq.orders {
        if order.ID == orderID {
            oq.orders = append(oq.orders[:i], oq.orders[i+1:]...)
            return true
        }
    }
    return false
}

type MatchingEngine struct {
    orderBooks    map[string]*OrderBook
    orders        map[string]*Order
    trades        []Trade
    buyQueues     map[string]*OrderQueue
    sellQueues    map[string]*OrderQueue
    mutex         sync.RWMutex
    orderChan     chan OrderRequest
    cancelChan    chan CancelRequest
    tradeChan     chan Trade
}

type OrderRequest struct {
    Order   Order
    Reply   chan OrderResponse
}

type OrderResponse struct {
    Order  *Order
    Trades []Trade
    Error  error
}

type CancelRequest struct {
    OrderID string
    UserID  string
    Reply   chan CancelResponse
}

type CancelResponse struct {
    Success bool
    Error   error
}

func NewMatchingEngine() *MatchingEngine {
    return &MatchingEngine{
        orderBooks: make(map[string]*OrderBook),
        orders:     make(map[string]*Order),
        trades:     make([]Trade, 0),
        buyQueues:  make(map[string]*OrderQueue),
        sellQueues: make(map[string]*OrderQueue),
        orderChan:  make(chan OrderRequest, 10000),
        cancelChan: make(chan CancelRequest, 1000),
        tradeChan:  make(chan Trade, 10000),
    }
}

func (me *MatchingEngine) SubmitOrder(order Order) (*Order, []Trade, error) {
    // Validate order
    if err := me.validateOrder(order); err != nil {
        return nil, nil, err
    }

    // Set order properties
    order.ID = uuid.New().String()
    order.Status = StatusPending
    order.FilledQuantity = 0
    order.RemainingQuantity = order.Quantity
    order.AveragePrice = 0
    order.CreatedAt = time.Now()
    order.UpdatedAt = time.Now()

    // Store order
    me.mutex.Lock()
    me.orders[order.ID] = &order
    me.mutex.Unlock()

    // Process order
    trades := me.processOrder(&order)

    // Update order status
    if order.RemainingQuantity == 0 {
        order.Status = StatusFilled
    } else if order.FilledQuantity > 0 {
        order.Status = StatusPartiallyFilled
    }

    order.UpdatedAt = time.Now()

    // Update order book
    me.updateOrderBook(order.Symbol)

    return &order, trades, nil
}

func (me *MatchingEngine) processOrder(order *Order) []Trade {
    var trades []Trade

    if order.Side == SideBuy {
        trades = me.matchBuyOrder(order)
    } else {
        trades = me.matchSellOrder(order)
    }

    // Add remaining order to queue if not fully filled
    if order.RemainingQuantity > 0 {
        me.addOrderToQueue(order)
    }

    return trades
}

func (me *MatchingEngine) matchBuyOrder(buyOrder *Order) []Trade {
    var trades []Trade
    symbol := buyOrder.Symbol

    // Get sell queue for symbol
    me.mutex.RLock()
    sellQueue, exists := me.sellQueues[symbol]
    me.mutex.RUnlock()

    if !exists {
        return trades
    }

    // Match against sell orders
    for buyOrder.RemainingQuantity > 0 {
        sellOrder := sellQueue.Peek()
        if sellOrder == nil {
            break
        }

        // Check if prices match
        if buyOrder.Price < sellOrder.Price {
            break
        }

        // Calculate trade quantity
        tradeQuantity := min(buyOrder.RemainingQuantity, sellOrder.RemainingQuantity)

        // Create trade
        trade := Trade{
            ID:          uuid.New().String(),
            BuyOrderID:  buyOrder.ID,
            SellOrderID: sellOrder.ID,
            Symbol:      symbol,
            Quantity:    tradeQuantity,
            Price:       sellOrder.Price,
            Timestamp:   time.Now(),
        }

        trades = append(trades, trade)

        // Update orders
        buyOrder.FilledQuantity += tradeQuantity
        buyOrder.RemainingQuantity -= tradeQuantity
        buyOrder.AveragePrice = me.calculateAveragePrice(buyOrder)

        sellOrder.FilledQuantity += tradeQuantity
        sellOrder.RemainingQuantity -= tradeQuantity
        sellOrder.AveragePrice = me.calculateAveragePrice(sellOrder)

        // Remove sell order if fully filled
        if sellOrder.RemainingQuantity == 0 {
            sellQueue.Pop()
            sellOrder.Status = StatusFilled
        }

        // Send trade to channel
        me.tradeChan <- trade
    }

    return trades
}

func (me *MatchingEngine) matchSellOrder(sellOrder *Order) []Trade {
    var trades []Trade
    symbol := sellOrder.Symbol

    // Get buy queue for symbol
    me.mutex.RLock()
    buyQueue, exists := me.buyQueues[symbol]
    me.mutex.RUnlock()

    if !exists {
        return trades
    }

    // Match against buy orders
    for sellOrder.RemainingQuantity > 0 {
        buyOrder := buyQueue.Peek()
        if buyOrder == nil {
            break
        }

        // Check if prices match
        if sellOrder.Price > buyOrder.Price {
            break
        }

        // Calculate trade quantity
        tradeQuantity := min(sellOrder.RemainingQuantity, buyOrder.RemainingQuantity)

        // Create trade
        trade := Trade{
            ID:          uuid.New().String(),
            BuyOrderID:  buyOrder.ID,
            SellOrderID: sellOrder.ID,
            Symbol:      symbol,
            Quantity:    tradeQuantity,
            Price:       buyOrder.Price,
            Timestamp:   time.Now(),
        }

        trades = append(trades, trade)

        // Update orders
        sellOrder.FilledQuantity += tradeQuantity
        sellOrder.RemainingQuantity -= tradeQuantity
        sellOrder.AveragePrice = me.calculateAveragePrice(sellOrder)

        buyOrder.FilledQuantity += tradeQuantity
        buyOrder.RemainingQuantity -= tradeQuantity
        buyOrder.AveragePrice = me.calculateAveragePrice(buyOrder)

        // Remove buy order if fully filled
        if buyOrder.RemainingQuantity == 0 {
            buyQueue.Pop()
            buyOrder.Status = StatusFilled
        }

        // Send trade to channel
        me.tradeChan <- trade
    }

    return trades
}

func (me *MatchingEngine) addOrderToQueue(order *Order) {
    symbol := order.Symbol

    me.mutex.Lock()
    defer me.mutex.Unlock()

    if order.Side == SideBuy {
        if me.buyQueues[symbol] == nil {
            me.buyQueues[symbol] = &OrderQueue{orders: make([]*Order, 0)}
        }
        heap.Push(me.buyQueues[symbol], order)
    } else {
        if me.sellQueues[symbol] == nil {
            me.sellQueues[symbol] = &OrderQueue{orders: make([]*Order, 0)}
        }
        heap.Push(me.sellQueues[symbol], order)
    }
}

func (me *MatchingEngine) CancelOrder(orderID, userID string) error {
    me.mutex.Lock()
    defer me.mutex.Unlock()

    order, exists := me.orders[orderID]
    if !exists {
        return fmt.Errorf("order not found")
    }

    if order.UserID != userID {
        return fmt.Errorf("unauthorized")
    }

    if order.Status == StatusFilled || order.Status == StatusCancelled {
        return fmt.Errorf("order cannot be cancelled")
    }

    // Remove from queue
    symbol := order.Symbol
    if order.Side == SideBuy {
        if buyQueue, exists := me.buyQueues[symbol]; exists {
            buyQueue.Remove(orderID)
        }
    } else {
        if sellQueue, exists := me.sellQueues[symbol]; exists {
            sellQueue.Remove(orderID)
        }
    }

    // Update order status
    order.Status = StatusCancelled
    order.UpdatedAt = time.Now()

    // Update order book
    me.updateOrderBook(symbol)

    return nil
}

func (me *MatchingEngine) GetOrder(orderID string) (*Order, error) {
    me.mutex.RLock()
    defer me.mutex.RUnlock()

    order, exists := me.orders[orderID]
    if !exists {
        return nil, fmt.Errorf("order not found")
    }

    return order, nil
}

func (me *MatchingEngine) GetOrderBook(symbol string) *OrderBook {
    me.mutex.RLock()
    defer me.mutex.RUnlock()

    orderBook, exists := me.orderBooks[symbol]
    if !exists {
        return &OrderBook{Symbol: symbol, Bids: []OrderLevel{}, Asks: []OrderLevel{}}
    }

    return orderBook
}

func (me *MatchingEngine) updateOrderBook(symbol string) {
    me.mutex.Lock()
    defer me.mutex.Unlock()

    orderBook := &OrderBook{
        Symbol:    symbol,
        Bids:      []OrderLevel{},
        Asks:      []OrderLevel{},
        Timestamp: time.Now(),
    }

    // Build bid levels
    if buyQueue, exists := me.buyQueues[symbol]; exists {
        priceMap := make(map[float64]int64)
        orderCount := make(map[float64]int)

        for _, order := range buyQueue.orders {
            if order.RemainingQuantity > 0 {
                priceMap[order.Price] += order.RemainingQuantity
                orderCount[order.Price]++
            }
        }

        for price, quantity := range priceMap {
            orderBook.Bids = append(orderBook.Bids, OrderLevel{
                Price:    price,
                Quantity: quantity,
                Orders:   orderCount[price],
            })
        }

        // Sort bids by price (descending)
        sort.Slice(orderBook.Bids, func(i, j int) bool {
            return orderBook.Bids[i].Price > orderBook.Bids[j].Price
        })
    }

    // Build ask levels
    if sellQueue, exists := me.sellQueues[symbol]; exists {
        priceMap := make(map[float64]int64)
        orderCount := make(map[float64]int)

        for _, order := range sellQueue.orders {
            if order.RemainingQuantity > 0 {
                priceMap[order.Price] += order.RemainingQuantity
                orderCount[order.Price]++
            }
        }

        for price, quantity := range priceMap {
            orderBook.Asks = append(orderBook.Asks, OrderLevel{
                Price:    price,
                Quantity: quantity,
                Orders:   orderCount[price],
            })
        }

        // Sort asks by price (ascending)
        sort.Slice(orderBook.Asks, func(i, j int) bool {
            return orderBook.Asks[i].Price < orderBook.Asks[j].Price
        })
    }

    me.orderBooks[symbol] = orderBook
}

func (me *MatchingEngine) calculateAveragePrice(order *Order) float64 {
    if order.FilledQuantity == 0 {
        return 0
    }

    // Simple average price calculation
    // In production, this would be more sophisticated
    return order.Price
}

func (me *MatchingEngine) validateOrder(order Order) error {
    if order.Quantity <= 0 {
        return fmt.Errorf("invalid quantity")
    }

    if order.Price <= 0 {
        return fmt.Errorf("invalid price")
    }

    if order.Symbol == "" {
        return fmt.Errorf("symbol required")
    }

    if order.UserID == "" {
        return fmt.Errorf("user ID required")
    }

    return nil
}

func (me *MatchingEngine) ProcessOrders() {
    for req := range me.orderChan {
        order, trades, err := me.SubmitOrder(req.Order)
        
        req.Reply <- OrderResponse{
            Order:  order,
            Trades: trades,
            Error:  err,
        }
    }
}

func (me *MatchingEngine) ProcessCancellations() {
    for req := range me.cancelChan {
        err := me.CancelOrder(req.OrderID, req.UserID)
        
        req.Reply <- CancelResponse{
            Success: err == nil,
            Error:   err,
        }
    }
}

func (me *MatchingEngine) ProcessTrades() {
    for trade := range me.tradeChan {
        me.mutex.Lock()
        me.trades = append(me.trades, trade)
        me.mutex.Unlock()
        
        log.Printf("Trade executed: %+v", trade)
    }
}

// HTTP Handlers
func (me *MatchingEngine) SubmitOrderHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    var order Order
    if err := json.NewDecoder(r.Body).Decode(&order); err != nil {
        http.Error(w, "Invalid request", http.StatusBadRequest)
        return
    }

    result, trades, err := me.SubmitOrder(order)
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    response := map[string]interface{}{
        "order":  result,
        "trades": trades,
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func (me *MatchingEngine) CancelOrderHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodDelete {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    orderID := r.URL.Path[len("/api/orders/"):len(r.URL.Path)-len("/cancel")]
    userID := r.URL.Query().Get("userID")

    if err := me.CancelOrder(orderID, userID); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    w.WriteHeader(http.StatusOK)
}

func (me *MatchingEngine) GetOrderHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodGet {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    orderID := r.URL.Path[len("/api/orders/"):]
    order, err := me.GetOrder(orderID)
    if err != nil {
        http.Error(w, err.Error(), http.StatusNotFound)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(order)
}

func (me *MatchingEngine) GetOrderBookHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodGet {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    symbol := r.URL.Path[len("/api/market/book/"):]
    orderBook := me.GetOrderBook(symbol)

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(orderBook)
}

func min(a, b int64) int64 {
    if a < b {
        return a
    }
    return b
}

func main() {
    engine := NewMatchingEngine()

    // Start background workers
    go engine.ProcessOrders()
    go engine.ProcessCancellations()
    go engine.ProcessTrades()

    // HTTP routes
    http.HandleFunc("/api/orders/submit", engine.SubmitOrderHandler)
    http.HandleFunc("/api/orders/", engine.CancelOrderHandler)
    http.HandleFunc("/api/orders/", engine.GetOrderHandler)
    http.HandleFunc("/api/market/book/", engine.GetOrderBookHandler)

    log.Println("Order matching engine starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

## Unit Tests

```go
func TestMatchingEngine_SubmitOrder(t *testing.T) {
    engine := NewMatchingEngine()

    order := Order{
        UserID:      "user123",
        Symbol:      "AAPL",
        Side:        SideBuy,
        Quantity:    100,
        Price:       150.50,
        OrderType:   TypeLimit,
        TimeInForce: TIFGTC,
    }

    result, trades, err := engine.SubmitOrder(order)
    if err != nil {
        t.Fatalf("SubmitOrder() error = %v", err)
    }

    if result.ID == "" {
        t.Error("SubmitOrder() did not generate order ID")
    }

    if result.Status != StatusPending {
        t.Errorf("SubmitOrder() status = %v, want %v", result.Status, StatusPending)
    }

    if len(trades) != 0 {
        t.Errorf("SubmitOrder() expected 0 trades, got %d", len(trades))
    }
}

func TestMatchingEngine_OrderMatching(t *testing.T) {
    engine := NewMatchingEngine()

    // Submit buy order
    buyOrder := Order{
        UserID:      "user1",
        Symbol:      "AAPL",
        Side:        SideBuy,
        Quantity:    100,
        Price:       150.50,
        OrderType:   TypeLimit,
        TimeInForce: TIFGTC,
    }

    buyResult, _, err := engine.SubmitOrder(buyOrder)
    if err != nil {
        t.Fatalf("SubmitOrder() error = %v", err)
    }

    // Submit sell order
    sellOrder := Order{
        UserID:      "user2",
        Symbol:      "AAPL",
        Side:        SideSell,
        Quantity:    100,
        Price:       150.50,
        OrderType:   TypeLimit,
        TimeInForce: TIFGTC,
    }

    sellResult, trades, err := engine.SubmitOrder(sellOrder)
    if err != nil {
        t.Fatalf("SubmitOrder() error = %v", err)
    }

    if len(trades) != 1 {
        t.Errorf("Expected 1 trade, got %d", len(trades))
    }

    if trades[0].Price != 150.50 {
        t.Errorf("Trade price = %v, want 150.50", trades[0].Price)
    }

    if buyResult.Status != StatusFilled {
        t.Errorf("Buy order status = %v, want %v", buyResult.Status, StatusFilled)
    }

    if sellResult.Status != StatusFilled {
        t.Errorf("Sell order status = %v, want %v", sellResult.Status, StatusFilled)
    }
}

func TestMatchingEngine_CancelOrder(t *testing.T) {
    engine := NewMatchingEngine()

    order := Order{
        UserID:      "user123",
        Symbol:      "AAPL",
        Side:        SideBuy,
        Quantity:    100,
        Price:       150.50,
        OrderType:   TypeLimit,
        TimeInForce: TIFGTC,
    }

    result, _, err := engine.SubmitOrder(order)
    if err != nil {
        t.Fatalf("SubmitOrder() error = %v", err)
    }

    err = engine.CancelOrder(result.ID, "user123")
    if err != nil {
        t.Fatalf("CancelOrder() error = %v", err)
    }

    updatedOrder, _ := engine.GetOrder(result.ID)
    if updatedOrder.Status != StatusCancelled {
        t.Errorf("Order status = %v, want %v", updatedOrder.Status, StatusCancelled)
    }
}
```

## Complexity Analysis

### Time Complexity
- **Submit Order**: O(log n) - Heap insertion
- **Cancel Order**: O(n) - Linear search in queue
- **Order Matching**: O(log n) - Heap operations
- **Order Book Update**: O(n) - Linear scan for price levels

### Space Complexity
- **Order Storage**: O(O) where O is number of orders
- **Trade Storage**: O(T) where T is number of trades
- **Order Book Storage**: O(L) where L is number of price levels
- **Total**: O(O + T + L)

## Edge Cases & Validation

### Input Validation
- Invalid order quantities (negative, zero)
- Invalid prices (negative, zero)
- Invalid symbols
- Invalid order types
- Invalid time in force

### Error Scenarios
- Order cancellation after fill
- Duplicate order IDs
- Insufficient balance
- Market hours validation
- Risk limit violations

### Boundary Conditions
- Maximum order quantity limits
- Price tick size validation
- Order book depth limits
- Trade size limits
- Time-based order expiration

## Extension Ideas (Scaling)

### Horizontal Scaling
1. **Load Balancing**: Multiple engine instances
2. **Symbol Sharding**: Partition by trading symbols
3. **Message Queue**: Kafka for order processing
4. **Database Sharding**: Partition by symbol or user

### Performance Optimization
1. **Lock-Free Data Structures**: Atomic operations for high performance
2. **Memory Pooling**: Reuse of order and trade objects
3. **Batch Processing**: Batch order processing
4. **CPU Affinity**: Bind threads to specific CPU cores

### Advanced Features
1. **Advanced Order Types**: Stop orders, trailing stops
2. **Risk Management**: Real-time position and risk monitoring
3. **Market Making**: Automated market making strategies
4. **Analytics**: Real-time trading analytics and reporting

## 20 Follow-up Questions

### 1. How would you handle order book reconstruction after system failure?
**Answer**: Implement order persistence with sequence numbers. Use event sourcing for order book reconstruction. Implement checkpoint mechanisms for fast recovery. Consider using write-ahead logging for durability.

### 2. What's your strategy for handling high-frequency trading?
**Answer**: Use lock-free data structures and atomic operations. Implement CPU affinity for thread binding. Use memory pools for object reuse. Consider using specialized hardware for ultra-low latency.

### 3. How do you ensure order matching fairness?
**Answer**: Implement strict price-time priority matching. Use deterministic order processing. Implement audit trails for order processing. Consider using consensus algorithms for distributed matching.

### 4. What's your approach to handling market data distribution?
**Answer**: Use multicast for market data distribution. Implement compression for market data. Use WebSocket for real-time updates. Consider using specialized market data protocols.

### 5. How would you implement risk management?
**Answer**: Implement real-time position monitoring. Use pre-trade risk checks. Implement position limits and exposure monitoring. Consider using machine learning for risk assessment.

### 6. What's your strategy for handling order book depth?
**Answer**: Implement configurable order book depth. Use efficient data structures for price levels. Implement order book compression. Consider using specialized order book data structures.

### 7. How do you handle partial order fills?
**Answer**: Implement partial fill tracking. Use average price calculation. Implement fill-or-kill and immediate-or-cancel logic. Consider using iceberg order support.

### 8. What's your approach to handling market hours?
**Answer**: Implement market hours validation. Use timezone-aware scheduling. Implement pre-market and after-hours handling. Consider using market calendar integration.

### 9. How would you implement order routing?
**Answer**: Implement smart order routing algorithms. Use latency-based routing. Implement venue selection logic. Consider using machine learning for routing optimization.

### 10. What's your strategy for handling order book imbalances?
**Answer**: Implement imbalance detection algorithms. Use market making strategies. Implement liquidity provision mechanisms. Consider using algorithmic trading strategies.

### 11. How do you handle order book snapshots?
**Answer**: Implement efficient order book snapshots. Use incremental updates for efficiency. Implement snapshot compression. Consider using specialized snapshot protocols.

### 12. What's your approach to handling order book updates?
**Answer**: Implement incremental order book updates. Use delta compression for updates. Implement update batching for efficiency. Consider using specialized update protocols.

### 13. How would you implement order book validation?
**Answer**: Implement order book integrity checks. Use checksums for validation. Implement cross-validation with multiple sources. Consider using consensus-based validation.

### 14. What's your strategy for handling order book latency?
**Answer**: Implement latency monitoring and measurement. Use low-latency data structures. Implement CPU optimization techniques. Consider using specialized hardware acceleration.

### 15. How do you handle order book scalability?
**Answer**: Implement horizontal scaling strategies. Use symbol-based sharding. Implement load balancing for order processing. Consider using distributed order book architectures.

### 16. What's your approach to handling order book persistence?
**Answer**: Implement efficient order book persistence. Use checkpoint mechanisms for recovery. Implement incremental persistence for efficiency. Consider using specialized persistence protocols.

### 17. How would you implement order book monitoring?
**Answer**: Implement real-time order book monitoring. Use metrics and alerting for anomalies. Implement performance monitoring. Consider using specialized monitoring tools.

### 18. What's your strategy for handling order book testing?
**Answer**: Implement comprehensive order book testing. Use simulation for testing. Implement stress testing for performance. Consider using specialized testing frameworks.

### 19. How do you handle order book security?
**Answer**: Implement access controls for order book data. Use encryption for sensitive data. Implement audit trails for security. Consider using specialized security protocols.

### 20. What's your approach to handling order book compliance?
**Answer**: Implement regulatory compliance monitoring. Use audit trails for compliance. Implement reporting for regulators. Consider using specialized compliance tools.

## Evaluation Checklist

### Code Quality (25%)
- [ ] Clean, readable Go code with proper error handling
- [ ] Appropriate use of interfaces and structs
- [ ] Proper concurrency patterns (goroutines, channels)
- [ ] Good separation of concerns

### Architecture (25%)
- [ ] Scalable design with efficient data structures
- [ ] Proper order matching algorithm
- [ ] Efficient order book management
- [ ] High-performance implementation

### Functionality (25%)
- [ ] Order submission and cancellation working
- [ ] Order matching algorithm functional
- [ ] Order book management working
- [ ] Trade execution implemented

### Testing (15%)
- [ ] Unit tests for core functionality
- [ ] Integration tests for API endpoints
- [ ] Edge case testing
- [ ] Performance testing

### Discussion (10%)
- [ ] Clear explanation of design decisions
- [ ] Understanding of trading systems
- [ ] Knowledge of order matching algorithms
- [ ] Ability to discuss trade-offs

## Discussion Pointers

### Key Points to Highlight
1. **Order Matching Algorithm**: Explain the price-time priority matching logic
2. **Data Structures**: Discuss the use of heaps for order queues
3. **Concurrency**: Explain the thread-safe order processing
4. **Performance**: Discuss the high-performance design considerations
5. **Order Book Management**: Explain the efficient order book representation

### Trade-offs to Discuss
1. **Latency vs Throughput**: Ultra-low latency vs high throughput trade-offs
2. **Consistency vs Performance**: Strong consistency vs high performance trade-offs
3. **Memory vs CPU**: Memory usage vs CPU optimization trade-offs
4. **Simplicity vs Features**: Simple design vs advanced features trade-offs
5. **Accuracy vs Speed**: Accurate matching vs fast processing trade-offs

### Extension Scenarios
1. **Multi-venue Trading**: How to handle multiple trading venues
2. **Advanced Order Types**: Complex order types and strategies
3. **Risk Management**: Real-time risk monitoring and control
4. **Market Data**: High-frequency market data distribution
5. **Compliance**: Regulatory compliance and reporting
