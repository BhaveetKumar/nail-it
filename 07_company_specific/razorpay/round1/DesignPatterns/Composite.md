---
# Auto-generated front matter
Title: Composite
LastUpdated: 2025-11-06T20:45:58.512084
Tags: []
Status: draft
---

# Composite Pattern

## Pattern Name & Intent

**Composite** is a structural design pattern that lets you compose objects into tree structures and then work with these structures as if they were individual objects. It allows clients to treat individual objects and compositions of objects uniformly.

**Key Intent:**

- Compose objects into tree structures to represent part-whole hierarchies
- Let clients treat individual objects and compositions uniformly
- Make it easier to add new kinds of components
- Provide a structure for building a hierarchy of objects
- Enable recursive composition of objects

## When to Use

**Use Composite when:**

1. **Tree Structures**: You need to represent part-whole hierarchies of objects
2. **Uniform Treatment**: Want clients to treat simple and complex elements uniformly
3. **Recursive Structures**: Need to work with tree-like object structures
4. **GUI Components**: Building user interface components with nested elements
5. **File Systems**: Representing files and directories
6. **Organizational Structures**: Modeling hierarchical organizations
7. **Mathematical Expressions**: Building expression trees

**Don't use when:**

- Object structure is not hierarchical
- You need type safety and want to distinguish between leaves and composites
- The hierarchy is very simple and doesn't justify the pattern complexity
- Performance is critical (adds overhead for uniform interface)

## Real-World Use Cases (Payments/Fintech)

### 1. Financial Portfolio Management

```go
// Portfolio components that can contain other components
type PortfolioComponent interface {
    GetValue() decimal.Decimal
    GetRisk() float64
    GetDescription() string
    AddComponent(component PortfolioComponent) error
    RemoveComponent(component PortfolioComponent) error
    GetComponents() []PortfolioComponent
    Accept(visitor PortfolioVisitor) error
}

// Individual stock (leaf)
type Stock struct {
    Symbol   string
    Shares   int64
    Price    decimal.Decimal
    Beta     float64 // Risk measure
}

func (s *Stock) GetValue() decimal.Decimal {
    return s.Price.Mul(decimal.NewFromInt(s.Shares))
}

func (s *Stock) GetRisk() float64 {
    return s.Beta
}

// Mutual fund (composite)
type MutualFund struct {
    Name       string
    Holdings   []PortfolioComponent
    ExpenseRatio float64
}

func (m *MutualFund) GetValue() decimal.Decimal {
    total := decimal.Zero
    for _, holding := range m.Holdings {
        total = total.Add(holding.GetValue())
    }
    return total
}

func (m *MutualFund) GetRisk() float64 {
    // Weighted average risk of holdings
    totalValue := m.GetValue()
    weightedRisk := 0.0

    for _, holding := range m.Holdings {
        weight := holding.GetValue().Div(totalValue).InexactFloat64()
        weightedRisk += weight * holding.GetRisk()
    }

    return weightedRisk
}

// Portfolio (composite)
type Portfolio struct {
    Name       string
    Components []PortfolioComponent
    Owner      string
}

func (p *Portfolio) GetValue() decimal.Decimal {
    total := decimal.Zero
    for _, component := range p.Components {
        total = total.Add(component.GetValue())
    }
    return total
}

func (p *Portfolio) AddComponent(component PortfolioComponent) error {
    p.Components = append(p.Components, component)
    return nil
}
```

### 2. Organizational Chart System

```go
// Organizational unit that can contain other units
type OrganizationalUnit interface {
    GetName() string
    GetBudget() decimal.Decimal
    GetEmployeeCount() int
    AddUnit(unit OrganizationalUnit) error
    RemoveUnit(unit OrganizationalUnit) error
    GetSubUnits() []OrganizationalUnit
    CalculateTotalCost() decimal.Decimal
}

// Employee (leaf)
type Employee struct {
    Name     string
    Position string
    Salary   decimal.Decimal
    Benefits decimal.Decimal
}

func (e *Employee) GetBudget() decimal.Decimal {
    return e.Salary.Add(e.Benefits)
}

func (e *Employee) GetEmployeeCount() int {
    return 1
}

// Department (composite)
type Department struct {
    Name      string
    Manager   *Employee
    SubUnits  []OrganizationalUnit
    Budget    decimal.Decimal
}

func (d *Department) GetBudget() decimal.Decimal {
    total := d.Budget
    for _, unit := range d.SubUnits {
        total = total.Add(unit.GetBudget())
    }
    return total
}

func (d *Department) GetEmployeeCount() int {
    count := 0
    if d.Manager != nil {
        count = 1
    }

    for _, unit := range d.SubUnits {
        count += unit.GetEmployeeCount()
    }

    return count
}

// Company (composite)
type Company struct {
    Name        string
    Departments []OrganizationalUnit
    CEO         *Employee
}

func (c *Company) GetBudget() decimal.Decimal {
    total := decimal.Zero
    if c.CEO != nil {
        total = total.Add(c.CEO.GetBudget())
    }

    for _, dept := range c.Departments {
        total = total.Add(dept.GetBudget())
    }

    return total
}
```

### 3. Transaction Processing Hierarchy

```go
// Transaction component that can be simple or composite
type TransactionComponent interface {
    Process() error
    GetAmount() decimal.Decimal
    GetFee() decimal.Decimal
    GetDescription() string
    AddTransaction(tx TransactionComponent) error
    RemoveTransaction(tx TransactionComponent) error
    GetTransactions() []TransactionComponent
    Validate() error
}

// Simple payment (leaf)
type SimplePayment struct {
    ID          string
    Amount      decimal.Decimal
    Fee         decimal.Decimal
    FromAccount string
    ToAccount   string
    Description string
}

func (s *SimplePayment) Process() error {
    // Process individual payment
    return processPayment(s.FromAccount, s.ToAccount, s.Amount)
}

func (s *SimplePayment) GetAmount() decimal.Decimal {
    return s.Amount
}

// Batch payment (composite)
type BatchPayment struct {
    ID           string
    Payments     []TransactionComponent
    Description  string
    BatchFee     decimal.Decimal
}

func (b *BatchPayment) Process() error {
    // Process all payments in batch
    for _, payment := range b.Payments {
        if err := payment.Process(); err != nil {
            return fmt.Errorf("batch payment failed: %w", err)
        }
    }
    return nil
}

func (b *BatchPayment) GetAmount() decimal.Decimal {
    total := decimal.Zero
    for _, payment := range b.Payments {
        total = total.Add(payment.GetAmount())
    }
    return total
}

func (b *BatchPayment) GetFee() decimal.Decimal {
    total := b.BatchFee
    for _, payment := range b.Payments {
        total = total.Add(payment.GetFee())
    }
    return total
}

// Split payment (composite)
type SplitPayment struct {
    ID          string
    Splits      []TransactionComponent
    TotalAmount decimal.Decimal
    Description string
}

func (s *SplitPayment) Process() error {
    // Validate splits add up to total
    splitTotal := decimal.Zero
    for _, split := range s.Splits {
        splitTotal = splitTotal.Add(split.GetAmount())
    }

    if !splitTotal.Equal(s.TotalAmount) {
        return fmt.Errorf("split amounts don't match total: %v vs %v", splitTotal, s.TotalAmount)
    }

    // Process all splits
    for _, split := range s.Splits {
        if err := split.Process(); err != nil {
            return fmt.Errorf("split payment failed: %w", err)
        }
    }

    return nil
}
```

### 4. Risk Management Hierarchy

```go
// Risk component that can be individual or composite
type RiskComponent interface {
    CalculateRisk() float64
    GetRiskType() string
    GetDescription() string
    AddComponent(component RiskComponent) error
    RemoveComponent(component RiskComponent) error
    GetComponents() []RiskComponent
    GenerateReport() RiskReport
}

// Market risk (leaf)
type MarketRisk struct {
    AssetClass string
    Exposure   decimal.Decimal
    Volatility float64
    Beta       float64
}

func (m *MarketRisk) CalculateRisk() float64 {
    // VaR calculation for market risk
    return m.Exposure.InexactFloat64() * m.Volatility * m.Beta
}

// Credit risk (leaf)
type CreditRisk struct {
    Counterparty    string
    Exposure        decimal.Decimal
    ProbabilityOfDefault float64
    LossGivenDefault     float64
}

func (c *CreditRisk) CalculateRisk() float64 {
    // Expected loss calculation
    return c.Exposure.InexactFloat64() * c.ProbabilityOfDefault * c.LossGivenDefault
}

// Portfolio risk (composite)
type PortfolioRisk struct {
    Name       string
    Risks      []RiskComponent
    Correlation map[string]map[string]float64 // Correlation matrix
}

func (p *PortfolioRisk) CalculateRisk() float64 {
    // Portfolio risk with correlation adjustments
    totalRisk := 0.0

    for i, risk1 := range p.Risks {
        for j, risk2 := range p.Risks {
            if i <= j {
                risk1Value := risk1.CalculateRisk()
                risk2Value := risk2.CalculateRisk()
                correlation := p.getCorrelation(risk1.GetRiskType(), risk2.GetRiskType())

                if i == j {
                    totalRisk += risk1Value * risk1Value
                } else {
                    totalRisk += 2 * risk1Value * risk2Value * correlation
                }
            }
        }
    }

    return math.Sqrt(totalRisk)
}
```

## Go Implementation

```go
package main

import (
    "fmt"
    "strings"
    "github.com/shopspring/decimal"
)

// Component interface
type FinancialComponent interface {
    GetValue() decimal.Decimal
    GetName() string
    GetType() string
    Display(indent int) string

    // Composite-specific methods
    Add(component FinancialComponent) error
    Remove(component FinancialComponent) error
    GetChild(index int) (FinancialComponent, error)
    GetChildren() []FinancialComponent
}

// Leaf: Individual Stock
type Stock struct {
    Symbol      string
    CompanyName string
    Shares      int64
    PricePerShare decimal.Decimal
    Currency    string
}

func NewStock(symbol, companyName string, shares int64, pricePerShare decimal.Decimal, currency string) *Stock {
    return &Stock{
        Symbol:        symbol,
        CompanyName:   companyName,
        Shares:        shares,
        PricePerShare: pricePerShare,
        Currency:      currency,
    }
}

func (s *Stock) GetValue() decimal.Decimal {
    return s.PricePerShare.Mul(decimal.NewFromInt(s.Shares))
}

func (s *Stock) GetName() string {
    return fmt.Sprintf("%s (%s)", s.CompanyName, s.Symbol)
}

func (s *Stock) GetType() string {
    return "Stock"
}

func (s *Stock) Display(indent int) string {
    indentStr := strings.Repeat("  ", indent)
    return fmt.Sprintf("%s📈 %s: %d shares @ %s %s = %s %s",
        indentStr, s.GetName(), s.Shares, s.PricePerShare, s.Currency, s.GetValue(), s.Currency)
}

// Leaf methods (not applicable for stocks)
func (s *Stock) Add(component FinancialComponent) error {
    return fmt.Errorf("cannot add components to a stock")
}

func (s *Stock) Remove(component FinancialComponent) error {
    return fmt.Errorf("cannot remove components from a stock")
}

func (s *Stock) GetChild(index int) (FinancialComponent, error) {
    return nil, fmt.Errorf("stock has no children")
}

func (s *Stock) GetChildren() []FinancialComponent {
    return nil
}

// Leaf: Bond
type Bond struct {
    ISIN         string
    IssuerName   string
    FaceValue    decimal.Decimal
    CouponRate   decimal.Decimal
    CurrentPrice decimal.Decimal
    Currency     string
}

func NewBond(isin, issuerName string, faceValue, couponRate, currentPrice decimal.Decimal, currency string) *Bond {
    return &Bond{
        ISIN:         isin,
        IssuerName:   issuerName,
        FaceValue:    faceValue,
        CouponRate:   couponRate,
        CurrentPrice: currentPrice,
        Currency:     currency,
    }
}

func (b *Bond) GetValue() decimal.Decimal {
    return b.CurrentPrice
}

func (b *Bond) GetName() string {
    return fmt.Sprintf("%s (%s)", b.IssuerName, b.ISIN)
}

func (b *Bond) GetType() string {
    return "Bond"
}

func (b *Bond) Display(indent int) string {
    indentStr := strings.Repeat("  ", indent)
    yield := b.CouponRate
    return fmt.Sprintf("%s🏛️  %s: %s %s (Yield: %s%%)",
        indentStr, b.GetName(), b.GetValue(), b.Currency, yield)
}

// Leaf methods (not applicable for bonds)
func (b *Bond) Add(component FinancialComponent) error {
    return fmt.Errorf("cannot add components to a bond")
}

func (b *Bond) Remove(component FinancialComponent) error {
    return fmt.Errorf("cannot remove components from a bond")
}

func (b *Bond) GetChild(index int) (FinancialComponent, error) {
    return nil, fmt.Errorf("bond has no children")
}

func (b *Bond) GetChildren() []FinancialComponent {
    return nil
}

// Composite: Portfolio
type Portfolio struct {
    Name       string
    Components []FinancialComponent
    Currency   string
}

func NewPortfolio(name, currency string) *Portfolio {
    return &Portfolio{
        Name:       name,
        Components: make([]FinancialComponent, 0),
        Currency:   currency,
    }
}

func (p *Portfolio) GetValue() decimal.Decimal {
    total := decimal.Zero
    for _, component := range p.Components {
        total = total.Add(component.GetValue())
    }
    return total
}

func (p *Portfolio) GetName() string {
    return p.Name
}

func (p *Portfolio) GetType() string {
    return "Portfolio"
}

func (p *Portfolio) Display(indent int) string {
    indentStr := strings.Repeat("  ", indent)
    result := fmt.Sprintf("%s📁 %s (Total: %s %s)",
        indentStr, p.Name, p.GetValue(), p.Currency)

    for _, component := range p.Components {
        result += "\n" + component.Display(indent+1)
    }

    return result
}

func (p *Portfolio) Add(component FinancialComponent) error {
    if component == nil {
        return fmt.Errorf("cannot add nil component")
    }

    p.Components = append(p.Components, component)
    return nil
}

func (p *Portfolio) Remove(component FinancialComponent) error {
    for i, comp := range p.Components {
        if comp == component {
            p.Components = append(p.Components[:i], p.Components[i+1:]...)
            return nil
        }
    }
    return fmt.Errorf("component not found in portfolio")
}

func (p *Portfolio) GetChild(index int) (FinancialComponent, error) {
    if index < 0 || index >= len(p.Components) {
        return nil, fmt.Errorf("index out of range: %d", index)
    }
    return p.Components[index], nil
}

func (p *Portfolio) GetChildren() []FinancialComponent {
    return p.Components
}

// Composite: Mutual Fund
type MutualFund struct {
    Name         string
    FundCode     string
    Holdings     []FinancialComponent
    ExpenseRatio decimal.Decimal
    Currency     string
    NAV          decimal.Decimal // Net Asset Value per unit
    UnitsHeld    decimal.Decimal
}

func NewMutualFund(name, fundCode string, expenseRatio, nav, unitsHeld decimal.Decimal, currency string) *MutualFund {
    return &MutualFund{
        Name:         name,
        FundCode:     fundCode,
        Holdings:     make([]FinancialComponent, 0),
        ExpenseRatio: expenseRatio,
        Currency:     currency,
        NAV:          nav,
        UnitsHeld:    unitsHeld,
    }
}

func (m *MutualFund) GetValue() decimal.Decimal {
    // Value based on NAV and units held
    grossValue := m.NAV.Mul(m.UnitsHeld)

    // Subtract expense ratio (simplified calculation)
    expenses := grossValue.Mul(m.ExpenseRatio).Div(decimal.NewFromInt(100))

    return grossValue.Sub(expenses)
}

func (m *MutualFund) GetName() string {
    return fmt.Sprintf("%s (%s)", m.Name, m.FundCode)
}

func (m *MutualFund) GetType() string {
    return "MutualFund"
}

func (m *MutualFund) Display(indent int) string {
    indentStr := strings.Repeat("  ", indent)
    result := fmt.Sprintf("%s🏦 %s: %s units @ %s %s = %s %s (Expense Ratio: %s%%)",
        indentStr, m.GetName(), m.UnitsHeld, m.NAV, m.Currency, m.GetValue(), m.Currency, m.ExpenseRatio)

    if len(m.Holdings) > 0 {
        result += "\n" + indentStr + "  Holdings:"
        for _, holding := range m.Holdings {
            result += "\n" + holding.Display(indent+2)
        }
    }

    return result
}

func (m *MutualFund) Add(component FinancialComponent) error {
    if component == nil {
        return fmt.Errorf("cannot add nil component")
    }

    m.Holdings = append(m.Holdings, component)
    return nil
}

func (m *MutualFund) Remove(component FinancialComponent) error {
    for i, holding := range m.Holdings {
        if holding == component {
            m.Holdings = append(m.Holdings[:i], m.Holdings[i+1:]...)
            return nil
        }
    }
    return fmt.Errorf("holding not found in mutual fund")
}

func (m *MutualFund) GetChild(index int) (FinancialComponent, error) {
    if index < 0 || index >= len(m.Holdings) {
        return nil, fmt.Errorf("index out of range: %d", index)
    }
    return m.Holdings[index], nil
}

func (m *MutualFund) GetChildren() []FinancialComponent {
    return m.Holdings
}

// Portfolio Analytics
type PortfolioAnalyzer struct{}

func (pa *PortfolioAnalyzer) CalculateAllocation(portfolio FinancialComponent) map[string]decimal.Decimal {
    allocation := make(map[string]decimal.Decimal)
    totalValue := portfolio.GetValue()

    pa.calculateAllocationRecursive(portfolio, allocation, totalValue)

    return allocation
}

func (pa *PortfolioAnalyzer) calculateAllocationRecursive(component FinancialComponent, allocation map[string]decimal.Decimal, totalValue decimal.Decimal) {
    componentType := component.GetType()
    componentValue := component.GetValue()

    if componentType == "Stock" || componentType == "Bond" {
        // Leaf component
        if existing, exists := allocation[componentType]; exists {
            allocation[componentType] = existing.Add(componentValue)
        } else {
            allocation[componentType] = componentValue
        }
    } else {
        // Composite component - recurse into children
        children := component.GetChildren()
        for _, child := range children {
            pa.calculateAllocationRecursive(child, allocation, totalValue)
        }
    }
}

func (pa *PortfolioAnalyzer) GetAllStocks(component FinancialComponent) []*Stock {
    var stocks []*Stock
    pa.collectStocks(component, &stocks)
    return stocks
}

func (pa *PortfolioAnalyzer) collectStocks(component FinancialComponent, stocks *[]*Stock) {
    if stock, ok := component.(*Stock); ok {
        *stocks = append(*stocks, stock)
    } else {
        children := component.GetChildren()
        for _, child := range children {
            pa.collectStocks(child, stocks)
        }
    }
}

func (pa *PortfolioAnalyzer) CountComponents(component FinancialComponent) map[string]int {
    counts := make(map[string]int)
    pa.countComponentsRecursive(component, counts)
    return counts
}

func (pa *PortfolioAnalyzer) countComponentsRecursive(component FinancialComponent, counts map[string]int) {
    componentType := component.GetType()
    counts[componentType]++

    children := component.GetChildren()
    for _, child := range children {
        pa.countComponentsRecursive(child, counts)
    }
}

// Visitor pattern integration
type PortfolioVisitor interface {
    VisitStock(stock *Stock) error
    VisitBond(bond *Bond) error
    VisitPortfolio(portfolio *Portfolio) error
    VisitMutualFund(fund *MutualFund) error
}

// Risk calculator visitor
type RiskCalculatorVisitor struct {
    TotalRisk    decimal.Decimal
    RiskByType   map[string]decimal.Decimal
    VolatilityMap map[string]decimal.Decimal
}

func NewRiskCalculatorVisitor() *RiskCalculatorVisitor {
    return &RiskCalculatorVisitor{
        TotalRisk:     decimal.Zero,
        RiskByType:    make(map[string]decimal.Decimal),
        VolatilityMap: make(map[string]decimal.Decimal),
    }
}

func (r *RiskCalculatorVisitor) VisitStock(stock *Stock) error {
    // Simplified risk calculation based on stock value
    volatility, exists := r.VolatilityMap[stock.Symbol]
    if !exists {
        volatility = decimal.NewFromFloat(0.15) // Default 15% volatility
    }

    stockRisk := stock.GetValue().Mul(volatility)
    r.TotalRisk = r.TotalRisk.Add(stockRisk)

    if existing, exists := r.RiskByType["Stock"]; exists {
        r.RiskByType["Stock"] = existing.Add(stockRisk)
    } else {
        r.RiskByType["Stock"] = stockRisk
    }

    return nil
}

func (r *RiskCalculatorVisitor) VisitBond(bond *Bond) error {
    // Bonds have lower risk - simplified calculation
    bondRisk := bond.GetValue().Mul(decimal.NewFromFloat(0.05)) // 5% risk
    r.TotalRisk = r.TotalRisk.Add(bondRisk)

    if existing, exists := r.RiskByType["Bond"]; exists {
        r.RiskByType["Bond"] = existing.Add(bondRisk)
    } else {
        r.RiskByType["Bond"] = bondRisk
    }

    return nil
}

func (r *RiskCalculatorVisitor) VisitPortfolio(portfolio *Portfolio) error {
    // Portfolio risk is calculated from its components
    for _, component := range portfolio.GetChildren() {
        if err := r.visitComponent(component); err != nil {
            return err
        }
    }
    return nil
}

func (r *RiskCalculatorVisitor) VisitMutualFund(fund *MutualFund) error {
    // Mutual fund risk based on expense ratio and holdings
    fundRisk := fund.GetValue().Mul(fund.ExpenseRatio.Div(decimal.NewFromInt(100)))
    r.TotalRisk = r.TotalRisk.Add(fundRisk)

    if existing, exists := r.RiskByType["MutualFund"]; exists {
        r.RiskByType["MutualFund"] = existing.Add(fundRisk)
    } else {
        r.RiskByType["MutualFund"] = fundRisk
    }

    // Visit holdings
    for _, holding := range fund.GetChildren() {
        if err := r.visitComponent(holding); err != nil {
            return err
        }
    }

    return nil
}

func (r *RiskCalculatorVisitor) visitComponent(component FinancialComponent) error {
    switch comp := component.(type) {
    case *Stock:
        return r.VisitStock(comp)
    case *Bond:
        return r.VisitBond(comp)
    case *Portfolio:
        return r.VisitPortfolio(comp)
    case *MutualFund:
        return r.VisitMutualFund(comp)
    default:
        return fmt.Errorf("unknown component type: %T", comp)
    }
}

// Example usage
func main() {
    fmt.Println("=== Composite Pattern Demo ===\n")

    // Create individual stocks (leaves)
    appleStock := NewStock("AAPL", "Apple Inc.", 100, decimal.NewFromFloat(150.00), "USD")
    googleStock := NewStock("GOOGL", "Alphabet Inc.", 50, decimal.NewFromFloat(2500.00), "USD")
    teslaStock := NewStock("TSLA", "Tesla Inc.", 75, decimal.NewFromFloat(800.00), "USD")

    // Create bonds (leaves)
    usTreasury := NewBond("US912810RZ35", "US Treasury", decimal.NewFromFloat(1000), decimal.NewFromFloat(2.5), decimal.NewFromFloat(980), "USD")
    corpBond := NewBond("XS1234567890", "Apple Corp Bond", decimal.NewFromFloat(1000), decimal.NewFromFloat(3.5), decimal.NewFromFloat(1020), "USD")

    // Create tech portfolio (composite)
    techPortfolio := NewPortfolio("Tech Stocks", "USD")
    techPortfolio.Add(appleStock)
    techPortfolio.Add(googleStock)
    techPortfolio.Add(teslaStock)

    // Create bond portfolio (composite)
    bondPortfolio := NewPortfolio("Fixed Income", "USD")
    bondPortfolio.Add(usTreasury)
    bondPortfolio.Add(corpBond)

    // Create mutual fund (composite)
    mutualFund := NewMutualFund("Vanguard S&P 500", "VFINX", decimal.NewFromFloat(0.14), decimal.NewFromFloat(350.00), decimal.NewFromFloat(100), "USD")
    mutualFund.Add(appleStock)
    mutualFund.Add(googleStock)

    // Create main portfolio (composite of composites)
    mainPortfolio := NewPortfolio("My Investment Portfolio", "USD")
    mainPortfolio.Add(techPortfolio)
    mainPortfolio.Add(bondPortfolio)
    mainPortfolio.Add(mutualFund)

    // Display the entire portfolio structure
    fmt.Println("Portfolio Structure:")
    fmt.Println(mainPortfolio.Display(0))

    fmt.Printf("\nTotal Portfolio Value: %s USD\n", mainPortfolio.GetValue())

    // Portfolio analysis
    fmt.Println("\n=== Portfolio Analysis ===")

    analyzer := &PortfolioAnalyzer{}

    // Asset allocation
    allocation := analyzer.CalculateAllocation(mainPortfolio)
    fmt.Println("\nAsset Allocation:")
    for assetType, value := range allocation {
        percentage := value.Div(mainPortfolio.GetValue()).Mul(decimal.NewFromInt(100))
        fmt.Printf("  %s: %s USD (%.2f%%)\n", assetType, value, percentage.InexactFloat64())
    }

    // Component counts
    counts := analyzer.CountComponents(mainPortfolio)
    fmt.Println("\nComponent Counts:")
    for componentType, count := range counts {
        fmt.Printf("  %s: %d\n", componentType, count)
    }

    // All stocks in portfolio
    allStocks := analyzer.GetAllStocks(mainPortfolio)
    fmt.Printf("\nAll Stocks in Portfolio (%d total):\n", len(allStocks))
    for _, stock := range allStocks {
        fmt.Printf("  %s: %s USD\n", stock.GetName(), stock.GetValue())
    }

    // Risk analysis using visitor pattern
    fmt.Println("\n=== Risk Analysis ===")

    riskCalculator := NewRiskCalculatorVisitor()

    // Set some volatility data
    riskCalculator.VolatilityMap["AAPL"] = decimal.NewFromFloat(0.20) // 20%
    riskCalculator.VolatilityMap["GOOGL"] = decimal.NewFromFloat(0.25) // 25%
    riskCalculator.VolatilityMap["TSLA"] = decimal.NewFromFloat(0.40) // 40%

    err := riskCalculator.visitComponent(mainPortfolio)
    if err != nil {
        fmt.Printf("Risk calculation error: %v\n", err)
    } else {
        fmt.Printf("Total Portfolio Risk: %s USD\n", riskCalculator.TotalRisk)
        fmt.Println("\nRisk by Asset Type:")
        for assetType, risk := range riskCalculator.RiskByType {
            fmt.Printf("  %s Risk: %s USD\n", assetType, risk)
        }
    }

    // Test composite operations
    fmt.Println("\n=== Composite Operations ===")

    // Add a new stock to tech portfolio
    microsoftStock := NewStock("MSFT", "Microsoft Corp.", 60, decimal.NewFromFloat(300.00), "USD")
    techPortfolio.Add(microsoftStock)
    fmt.Printf("Added Microsoft stock. Tech portfolio value: %s USD\n", techPortfolio.GetValue())

    // Remove a stock from tech portfolio
    techPortfolio.Remove(teslaStock)
    fmt.Printf("Removed Tesla stock. Tech portfolio value: %s USD\n", techPortfolio.GetValue())

    // Access child components
    child, err := techPortfolio.GetChild(0)
    if err != nil {
        fmt.Printf("Error accessing child: %v\n", err)
    } else {
        fmt.Printf("First child in tech portfolio: %s (Value: %s USD)\n", child.GetName(), child.GetValue())
    }

    fmt.Printf("\nFinal Portfolio Value: %s USD\n", mainPortfolio.GetValue())

    fmt.Println("\n=== Composite Pattern Demo Complete ===")
}
```

## Variants & Trade-offs

### Variants

1. **Strict Composite (Type Safety)**

```go
type Component interface {
    Operation() string
}

type Leaf struct {
    name string
}

func (l *Leaf) Operation() string {
    return l.name
}

type Composite struct {
    children []Component
}

func (c *Composite) Operation() string {
    result := "Composite["
    for i, child := range c.children {
        if i > 0 {
            result += ", "
        }
        result += child.Operation()
    }
    result += "]"
    return result
}

func (c *Composite) Add(component Component) {
    c.children = append(c.children, component)
}
```

2. **Safe Composite (Interface Segregation)**

```go
type Component interface {
    Operation() string
}

type CompositeOperations interface {
    Add(component Component)
    Remove(component Component)
    GetChild(index int) Component
}

type SafeComposite interface {
    Component
    CompositeOperations
}
```

3. **Cached Composite**

```go
type CachedComposite struct {
    children    []Component
    cachedValue *string
    dirty       bool
}

func (c *CachedComposite) Operation() string {
    if c.dirty || c.cachedValue == nil {
        result := "Composite["
        for i, child := range c.children {
            if i > 0 {
                result += ", "
            }
            result += child.Operation()
        }
        result += "]"
        c.cachedValue = &result
        c.dirty = false
    }
    return *c.cachedValue
}

func (c *CachedComposite) Add(component Component) {
    c.children = append(c.children, component)
    c.dirty = true
}
```

4. **Iterator Integration**

```go
type IterableComposite struct {
    *Composite
}

func (i *IterableComposite) Iterator() ComponentIterator {
    return NewComponentIterator(i.children)
}

type ComponentIterator struct {
    components []Component
    current    int
}

func (ci *ComponentIterator) HasNext() bool {
    return ci.current < len(ci.components)
}

func (ci *ComponentIterator) Next() Component {
    if !ci.HasNext() {
        return nil
    }
    component := ci.components[ci.current]
    ci.current++
    return component
}
```

### Trade-offs

**Pros:**

- **Uniform Treatment**: Clients can treat simple and complex objects uniformly
- **Easy Extension**: Easy to add new component types
- **Recursive Composition**: Natural support for tree structures
- **Flexibility**: Can build complex structures from simple parts
- **Simplified Client Code**: Clients don't need to distinguish between leaf and composite

**Cons:**

- **Type Safety**: May sacrifice type safety for uniformity
- **Design Constraint**: Can make the design overly general
- **Component Interface**: Interface may become too broad
- **Performance**: Recursive operations can be expensive
- **Memory Overhead**: Extra overhead for maintaining tree structure

**When to Choose Composite vs Alternatives:**

| Scenario                | Pattern                 | Reason                      |
| ----------------------- | ----------------------- | --------------------------- |
| Tree structures         | Composite               | Natural fit for hierarchies |
| Simple collections      | List/Array              | Less overhead               |
| Different behaviors     | Strategy                | Different algorithms        |
| Chain of responsibility | Chain of Responsibility | Sequential processing       |
| Observer pattern        | Observer                | Event notification          |

## Testable Example

```go
package main

import (
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "github.com/shopspring/decimal"
)

func TestStock_BasicOperations(t *testing.T) {
    stock := NewStock("AAPL", "Apple Inc.", 100, decimal.NewFromFloat(150.00), "USD")

    assert.Equal(t, "Apple Inc. (AAPL)", stock.GetName())
    assert.Equal(t, "Stock", stock.GetType())
    assert.Equal(t, decimal.NewFromFloat(15000.00), stock.GetValue()) // 100 * 150
    assert.Nil(t, stock.GetChildren())

    // Test that leaf operations return errors
    err := stock.Add(nil)
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "cannot add components to a stock")

    err = stock.Remove(nil)
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "cannot remove components from a stock")

    _, err = stock.GetChild(0)
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "stock has no children")
}

func TestBond_BasicOperations(t *testing.T) {
    bond := NewBond("US912810RZ35", "US Treasury", decimal.NewFromFloat(1000), decimal.NewFromFloat(2.5), decimal.NewFromFloat(980), "USD")

    assert.Equal(t, "US Treasury (US912810RZ35)", bond.GetName())
    assert.Equal(t, "Bond", bond.GetType())
    assert.Equal(t, decimal.NewFromFloat(980.00), bond.GetValue())
    assert.Nil(t, bond.GetChildren())

    // Test that leaf operations return errors
    err := bond.Add(nil)
    assert.Error(t, err)

    err = bond.Remove(nil)
    assert.Error(t, err)

    _, err = bond.GetChild(0)
    assert.Error(t, err)
}

func TestPortfolio_AddRemoveOperations(t *testing.T) {
    portfolio := NewPortfolio("Test Portfolio", "USD")
    stock1 := NewStock("AAPL", "Apple Inc.", 100, decimal.NewFromFloat(150.00), "USD")
    stock2 := NewStock("GOOGL", "Alphabet Inc.", 50, decimal.NewFromFloat(2500.00), "USD")

    // Test adding components
    err := portfolio.Add(stock1)
    assert.NoError(t, err)
    assert.Len(t, portfolio.GetChildren(), 1)

    err = portfolio.Add(stock2)
    assert.NoError(t, err)
    assert.Len(t, portfolio.GetChildren(), 2)

    // Test portfolio value calculation
    expectedValue := decimal.NewFromFloat(15000.00).Add(decimal.NewFromFloat(125000.00)) // 15k + 125k
    assert.Equal(t, expectedValue, portfolio.GetValue())

    // Test removing components
    err = portfolio.Remove(stock1)
    assert.NoError(t, err)
    assert.Len(t, portfolio.GetChildren(), 1)

    // Test removing non-existent component
    err = portfolio.Remove(stock1)
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "component not found")

    // Test adding nil component
    err = portfolio.Add(nil)
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "cannot add nil component")
}

func TestPortfolio_ChildAccess(t *testing.T) {
    portfolio := NewPortfolio("Test Portfolio", "USD")
    stock1 := NewStock("AAPL", "Apple Inc.", 100, decimal.NewFromFloat(150.00), "USD")
    stock2 := NewStock("GOOGL", "Alphabet Inc.", 50, decimal.NewFromFloat(2500.00), "USD")

    portfolio.Add(stock1)
    portfolio.Add(stock2)

    // Test valid child access
    child, err := portfolio.GetChild(0)
    assert.NoError(t, err)
    assert.Equal(t, stock1, child)

    child, err = portfolio.GetChild(1)
    assert.NoError(t, err)
    assert.Equal(t, stock2, child)

    // Test invalid child access
    _, err = portfolio.GetChild(2)
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "index out of range")

    _, err = portfolio.GetChild(-1)
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "index out of range")

    // Test GetChildren
    children := portfolio.GetChildren()
    assert.Len(t, children, 2)
    assert.Equal(t, stock1, children[0])
    assert.Equal(t, stock2, children[1])
}

func TestMutualFund_Operations(t *testing.T) {
    fund := NewMutualFund("Test Fund", "TFUND", decimal.NewFromFloat(0.5), decimal.NewFromFloat(100.00), decimal.NewFromFloat(50), "USD")

    assert.Equal(t, "Test Fund (TFUND)", fund.GetName())
    assert.Equal(t, "MutualFund", fund.GetType())

    // Test value calculation with expense ratio
    // Gross value: 100 * 50 = 5000
    // Expenses: 5000 * 0.5% = 25
    // Net value: 5000 - 25 = 4975
    expectedValue := decimal.NewFromFloat(4975.00)
    assert.Equal(t, expectedValue, fund.GetValue())

    // Test adding holdings
    stock := NewStock("AAPL", "Apple Inc.", 100, decimal.NewFromFloat(150.00), "USD")
    err := fund.Add(stock)
    assert.NoError(t, err)
    assert.Len(t, fund.GetChildren(), 1)

    // Test removing holdings
    err = fund.Remove(stock)
    assert.NoError(t, err)
    assert.Len(t, fund.GetChildren(), 0)
}

func TestCompositeHierarchy(t *testing.T) {
    // Create a complex hierarchy: Portfolio -> MutualFund -> Stocks
    mainPortfolio := NewPortfolio("Main Portfolio", "USD")

    // Create mutual fund with holdings
    fund := NewMutualFund("Tech Fund", "TECH", decimal.NewFromFloat(1.0), decimal.NewFromFloat(200.00), decimal.NewFromFloat(25), "USD")
    stock1 := NewStock("AAPL", "Apple Inc.", 100, decimal.NewFromFloat(150.00), "USD")
    stock2 := NewStock("GOOGL", "Alphabet Inc.", 50, decimal.NewFromFloat(2500.00), "USD")

    fund.Add(stock1)
    fund.Add(stock2)

    // Add fund to main portfolio
    mainPortfolio.Add(fund)

    // Add direct stock to main portfolio
    directStock := NewStock("MSFT", "Microsoft Corp.", 75, decimal.NewFromFloat(300.00), "USD")
    mainPortfolio.Add(directStock)

    // Test hierarchy value calculation
    fundValue := decimal.NewFromFloat(4950.00) // (200 * 25) - (5000 * 1%) = 5000 - 50 = 4950
    directStockValue := decimal.NewFromFloat(22500.00) // 75 * 300
    expectedTotal := fundValue.Add(directStockValue)

    assert.Equal(t, expectedTotal, mainPortfolio.GetValue())

    // Test hierarchy structure
    assert.Len(t, mainPortfolio.GetChildren(), 2)
    assert.Len(t, fund.GetChildren(), 2)
    assert.Len(t, directStock.GetChildren(), 0)
}

func TestPortfolioAnalyzer_CalculateAllocation(t *testing.T) {
    analyzer := &PortfolioAnalyzer{}

    portfolio := NewPortfolio("Test Portfolio", "USD")
    stock := NewStock("AAPL", "Apple Inc.", 100, decimal.NewFromFloat(150.00), "USD")
    bond := NewBond("US912810RZ35", "US Treasury", decimal.NewFromFloat(1000), decimal.NewFromFloat(2.5), decimal.NewFromFloat(1000), "USD")

    portfolio.Add(stock)
    portfolio.Add(bond)

    allocation := analyzer.CalculateAllocation(portfolio)

    assert.Len(t, allocation, 2)
    assert.Equal(t, decimal.NewFromFloat(15000.00), allocation["Stock"])
    assert.Equal(t, decimal.NewFromFloat(1000.00), allocation["Bond"])
}

func TestPortfolioAnalyzer_GetAllStocks(t *testing.T) {
    analyzer := &PortfolioAnalyzer{}

    portfolio := NewPortfolio("Test Portfolio", "USD")
    stock1 := NewStock("AAPL", "Apple Inc.", 100, decimal.NewFromFloat(150.00), "USD")
    stock2 := NewStock("GOOGL", "Alphabet Inc.", 50, decimal.NewFromFloat(2500.00), "USD")
    bond := NewBond("US912810RZ35", "US Treasury", decimal.NewFromFloat(1000), decimal.NewFromFloat(2.5), decimal.NewFromFloat(1000), "USD")

    // Create nested structure
    subPortfolio := NewPortfolio("Sub Portfolio", "USD")
    subPortfolio.Add(stock2)

    portfolio.Add(stock1)
    portfolio.Add(bond)
    portfolio.Add(subPortfolio)

    stocks := analyzer.GetAllStocks(portfolio)

    assert.Len(t, stocks, 2)
    assert.Contains(t, stocks, stock1)
    assert.Contains(t, stocks, stock2)
}

func TestPortfolioAnalyzer_CountComponents(t *testing.T) {
    analyzer := &PortfolioAnalyzer{}

    portfolio := NewPortfolio("Test Portfolio", "USD")
    stock := NewStock("AAPL", "Apple Inc.", 100, decimal.NewFromFloat(150.00), "USD")
    bond := NewBond("US912810RZ35", "US Treasury", decimal.NewFromFloat(1000), decimal.NewFromFloat(2.5), decimal.NewFromFloat(1000), "USD")
    fund := NewMutualFund("Test Fund", "TFUND", decimal.NewFromFloat(0.5), decimal.NewFromFloat(100.00), decimal.NewFromFloat(50), "USD")

    portfolio.Add(stock)
    portfolio.Add(bond)
    portfolio.Add(fund)

    counts := analyzer.CountComponents(portfolio)

    assert.Equal(t, 1, counts["Portfolio"])
    assert.Equal(t, 1, counts["Stock"])
    assert.Equal(t, 1, counts["Bond"])
    assert.Equal(t, 1, counts["MutualFund"])
}

func TestRiskCalculatorVisitor(t *testing.T) {
    calculator := NewRiskCalculatorVisitor()
    calculator.VolatilityMap["AAPL"] = decimal.NewFromFloat(0.20)

    portfolio := NewPortfolio("Test Portfolio", "USD")
    stock := NewStock("AAPL", "Apple Inc.", 100, decimal.NewFromFloat(150.00), "USD") // Value: 15000
    bond := NewBond("US912810RZ35", "US Treasury", decimal.NewFromFloat(1000), decimal.NewFromFloat(2.5), decimal.NewFromFloat(1000), "USD") // Value: 1000

    portfolio.Add(stock)
    portfolio.Add(bond)

    err := calculator.visitComponent(portfolio)
    assert.NoError(t, err)

    // Expected risks:
    // Stock: 15000 * 0.20 = 3000
    // Bond: 1000 * 0.05 = 50
    // Total: 3050

    expectedStockRisk := decimal.NewFromFloat(3000.00)
    expectedBondRisk := decimal.NewFromFloat(50.00)
    expectedTotalRisk := decimal.NewFromFloat(3050.00)

    assert.Equal(t, expectedTotalRisk, calculator.TotalRisk)
    assert.Equal(t, expectedStockRisk, calculator.RiskByType["Stock"])
    assert.Equal(t, expectedBondRisk, calculator.RiskByType["Bond"])
}

func TestPortfolioDisplay(t *testing.T) {
    portfolio := NewPortfolio("Test Portfolio", "USD")
    stock := NewStock("AAPL", "Apple Inc.", 100, decimal.NewFromFloat(150.00), "USD")
    portfolio.Add(stock)

    display := portfolio.Display(0)

    assert.Contains(t, display, "Test Portfolio")
    assert.Contains(t, display, "15000")
    assert.Contains(t, display, "Apple Inc.")
    assert.Contains(t, display, "📁") // Portfolio icon
    assert.Contains(t, display, "📈") // Stock icon
}

func BenchmarkPortfolioValueCalculation(b *testing.B) {
    portfolio := NewPortfolio("Benchmark Portfolio", "USD")

    // Add 1000 stocks to portfolio
    for i := 0; i < 1000; i++ {
        stock := NewStock(fmt.Sprintf("STOCK%d", i), fmt.Sprintf("Company %d", i), 100, decimal.NewFromFloat(150.00), "USD")
        portfolio.Add(stock)
    }

    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        portfolio.GetValue()
    }
}

func BenchmarkPortfolioAnalyzer_GetAllStocks(b *testing.B) {
    analyzer := &PortfolioAnalyzer{}

    portfolio := NewPortfolio("Benchmark Portfolio", "USD")

    // Create nested structure with 100 stocks
    for i := 0; i < 100; i++ {
        subPortfolio := NewPortfolio(fmt.Sprintf("Sub Portfolio %d", i), "USD")
        stock := NewStock(fmt.Sprintf("STOCK%d", i), fmt.Sprintf("Company %d", i), 100, decimal.NewFromFloat(150.00), "USD")
        subPortfolio.Add(stock)
        portfolio.Add(subPortfolio)
    }

    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        analyzer.GetAllStocks(portfolio)
    }
}
```

## Integration Tips

### 1. Persistence Integration

```go
type PersistableComponent interface {
    FinancialComponent
    Save() error
    Load(id string) error
    GetID() string
}

type PersistablePortfolio struct {
    *Portfolio
    ID string
    db Database
}

func (p *PersistablePortfolio) Save() error {
    return p.db.SavePortfolio(p.ID, p.Portfolio)
}

func (p *PersistablePortfolio) Load(id string) error {
    portfolio, err := p.db.LoadPortfolio(id)
    if err != nil {
        return err
    }
    p.Portfolio = portfolio
    return nil
}
```

### 2. Observer Integration

```go
type ObservableComponent interface {
    FinancialComponent
    AddObserver(observer ComponentObserver)
    RemoveObserver(observer ComponentObserver)
    NotifyObservers(event ComponentEvent)
}

type ComponentObserver interface {
    OnComponentChanged(component FinancialComponent, event ComponentEvent)
}

type ObservablePortfolio struct {
    *Portfolio
    observers []ComponentObserver
}

func (o *ObservablePortfolio) Add(component FinancialComponent) error {
    err := o.Portfolio.Add(component)
    if err == nil {
        o.NotifyObservers(ComponentAddedEvent{Component: component})
    }
    return err
}
```

### 3. Strategy Integration

```go
type ValueCalculationStrategy interface {
    CalculateValue(component FinancialComponent) decimal.Decimal
}

type MarketValueStrategy struct{}

func (m *MarketValueStrategy) CalculateValue(component FinancialComponent) decimal.Decimal {
    return component.GetValue()
}

type BookValueStrategy struct{}

func (b *BookValueStrategy) CalculateValue(component FinancialComponent) decimal.Decimal {
    // Calculate book value based on different criteria
    return component.GetValue().Mul(decimal.NewFromFloat(0.8))
}

type StrategicPortfolio struct {
    *Portfolio
    strategy ValueCalculationStrategy
}

func (s *StrategicPortfolio) GetValue() decimal.Decimal {
    return s.strategy.CalculateValue(s.Portfolio)
}
```

### 4. Factory Integration

```go
type ComponentFactory interface {
    CreateStock(symbol, name string, shares int64, price decimal.Decimal) FinancialComponent
    CreateBond(isin, issuer string, faceValue, couponRate, currentPrice decimal.Decimal) FinancialComponent
    CreatePortfolio(name string) FinancialComponent
}

type DefaultComponentFactory struct{}

func (d *DefaultComponentFactory) CreateStock(symbol, name string, shares int64, price decimal.Decimal) FinancialComponent {
    return NewStock(symbol, name, shares, price, "USD")
}

func (d *DefaultComponentFactory) CreatePortfolio(name string) FinancialComponent {
    return NewPortfolio(name, "USD")
}

type PortfolioBuilder struct {
    factory ComponentFactory
    portfolio FinancialComponent
}

func (p *PortfolioBuilder) AddStock(symbol, name string, shares int64, price decimal.Decimal) *PortfolioBuilder {
    stock := p.factory.CreateStock(symbol, name, shares, price)
    p.portfolio.Add(stock)
    return p
}
```

## Common Interview Questions

### 1. **How does Composite pattern achieve uniform treatment of objects?**

**Answer:**
Composite pattern achieves uniform treatment through a common interface that both leaf and composite objects implement:

```go
type Component interface {
    Operation() string
    Add(Component) error    // May not apply to leaves
    Remove(Component) error // May not apply to leaves
    GetChild(int) (Component, error) // May not apply to leaves
}

// Client code treats both uniformly
func ProcessComponent(comp Component) {
    result := comp.Operation() // Works for both leaf and composite
    fmt.Println(result)

    // Optional: Handle composites
    if children := comp.GetChildren(); children != nil {
        for _, child := range children {
            ProcessComponent(child) // Recursive processing
        }
    }
}

// Usage - same interface for different types
var leaf Component = &Stock{...}
var composite Component = &Portfolio{...}

ProcessComponent(leaf)      // Works
ProcessComponent(composite) // Also works
```

**Benefits:**

- **Client Simplicity**: Clients don't need to distinguish between types
- **Polymorphism**: Same operations work on different object types
- **Recursive Operations**: Natural support for tree traversal
- **Extensibility**: Easy to add new component types

### 2. **How do you handle operations that don't apply to leaf nodes?**

**Answer:**
There are several approaches to handle operations that don't apply to leaves:

**Approach 1: Return Error for Inappropriate Operations**

```go
type Stock struct {
    // ... fields
}

func (s *Stock) Add(component Component) error {
    return fmt.Errorf("cannot add components to a stock (leaf node)")
}

func (s *Stock) Remove(component Component) error {
    return fmt.Errorf("cannot remove components from a stock (leaf node)")
}

func (s *Stock) GetChild(index int) (Component, error) {
    return nil, fmt.Errorf("stock has no children")
}
```

**Approach 2: Default Implementation (No-op)**

```go
type Stock struct {
    // ... fields
}

func (s *Stock) Add(component Component) error {
    // Silently ignore - no-op
    return nil
}

func (s *Stock) GetChildren() []Component {
    return nil // Return empty slice
}
```

**Approach 3: Interface Segregation**

```go
type Component interface {
    Operation() string
    GetValue() decimal.Decimal
}

type Composite interface {
    Component
    Add(component Component) error
    Remove(component Component) error
    GetChildren() []Component
}

// Type assertion for composite operations
func AddToComposite(comp Component, child Component) error {
    if composite, ok := comp.(Composite); ok {
        return composite.Add(child)
    }
    return fmt.Errorf("component does not support child operations")
}
```

**Best Practice**: Use approach 1 (return errors) for clarity and explicit error handling.

### 3. **How do you implement efficient tree traversal in Composite pattern?**

**Answer:**
Implement various traversal strategies based on needs:

**1. Depth-First Traversal (Recursive)**

```go
func (p *Portfolio) TraverseDepthFirst(visitor func(Component) error) error {
    // Visit current node
    if err := visitor(p); err != nil {
        return err
    }

    // Visit children
    for _, child := range p.GetChildren() {
        if composite, ok := child.(interface{ TraverseDepthFirst(func(Component) error) error }); ok {
            if err := composite.TraverseDepthFirst(visitor); err != nil {
                return err
            }
        } else {
            if err := visitor(child); err != nil {
                return err
            }
        }
    }

    return nil
}
```

**2. Breadth-First Traversal (Iterative)**

```go
func (p *Portfolio) TraverseBreadthFirst(visitor func(Component) error) error {
    queue := []Component{p}

    for len(queue) > 0 {
        current := queue[0]
        queue = queue[1:]

        if err := visitor(current); err != nil {
            return err
        }

        // Add children to queue
        if children := current.GetChildren(); children != nil {
            queue = append(queue, children...)
        }
    }

    return nil
}
```

**3. Iterator Pattern Integration**

```go
type ComponentIterator interface {
    HasNext() bool
    Next() Component
}

type DepthFirstIterator struct {
    stack []Component
}

func (d *DepthFirstIterator) HasNext() bool {
    return len(d.stack) > 0
}

func (d *DepthFirstIterator) Next() Component {
    if !d.HasNext() {
        return nil
    }

    current := d.stack[len(d.stack)-1]
    d.stack = d.stack[:len(d.stack)-1]

    // Add children to stack (reverse order for DFS)
    if children := current.GetChildren(); children != nil {
        for i := len(children) - 1; i >= 0; i-- {
            d.stack = append(d.stack, children[i])
        }
    }

    return current
}

func (p *Portfolio) Iterator() ComponentIterator {
    return &DepthFirstIterator{
        stack: []Component{p},
    }
}
```

**4. Visitor Pattern for Complex Operations**

```go
type ComponentVisitor interface {
    VisitStock(stock *Stock) error
    VisitBond(bond *Bond) error
    VisitPortfolio(portfolio *Portfolio) error
}

func (p *Portfolio) Accept(visitor ComponentVisitor) error {
    if err := visitor.VisitPortfolio(p); err != nil {
        return err
    }

    for _, child := range p.GetChildren() {
        if err := child.Accept(visitor); err != nil {
            return err
        }
    }

    return nil
}
```

### 4. **How do you handle cycles in Composite structures?**

**Answer:**
Prevent and detect cycles using several strategies:

**1. Cycle Detection during Addition**

```go
func (p *Portfolio) Add(component FinancialComponent) error {
    if component == nil {
        return fmt.Errorf("cannot add nil component")
    }

    // Check for cycles
    if p.createsCycle(component) {
        return fmt.Errorf("adding component would create a cycle")
    }

    p.Components = append(p.Components, component)
    return nil
}

func (p *Portfolio) createsCycle(component FinancialComponent) bool {
    // Use DFS to check if component contains this portfolio
    visited := make(map[FinancialComponent]bool)
    return p.hasCycle(component, visited)
}

func (p *Portfolio) hasCycle(component FinancialComponent, visited map[FinancialComponent]bool) bool {
    if component == p {
        return true // Found cycle
    }

    if visited[component] {
        return false // Already visited, no cycle through this path
    }

    visited[component] = true

    children := component.GetChildren()
    if children != nil {
        for _, child := range children {
            if p.hasCycle(child, visited) {
                return true
            }
        }
    }

    return false
}
```

**2. Parent References**

```go
type ComponentWithParent interface {
    FinancialComponent
    SetParent(parent FinancialComponent)
    GetParent() FinancialComponent
}

type SafePortfolio struct {
    *Portfolio
    parent FinancialComponent
}

func (s *SafePortfolio) Add(component FinancialComponent) error {
    // Check if component is an ancestor
    if s.isAncestor(component) {
        return fmt.Errorf("cannot add ancestor as child - would create cycle")
    }

    if err := s.Portfolio.Add(component); err != nil {
        return err
    }

    // Set parent reference
    if parentAware, ok := component.(ComponentWithParent); ok {
        parentAware.SetParent(s)
    }

    return nil
}

func (s *SafePortfolio) isAncestor(component FinancialComponent) bool {
    current := s.parent
    for current != nil {
        if current == component {
            return true
        }
        if parentAware, ok := current.(ComponentWithParent); ok {
            current = parentAware.GetParent()
        } else {
            break
        }
    }
    return false
}
```

**3. Immutable Structure Approach**

```go
type ImmutablePortfolio struct {
    name       string
    components []FinancialComponent
    value      decimal.Decimal
}

func (i *ImmutablePortfolio) WithComponent(component FinancialComponent) (*ImmutablePortfolio, error) {
    // Create new portfolio with additional component
    newComponents := make([]FinancialComponent, len(i.components)+1)
    copy(newComponents, i.components)
    newComponents[len(i.components)] = component

    return &ImmutablePortfolio{
        name:       i.name,
        components: newComponents,
        value:      i.value.Add(component.GetValue()),
    }, nil
}
```

### 5. **How do you implement caching in Composite pattern for performance?**

**Answer:**
Implement caching at different levels:

**1. Cached Value Calculation**

```go
type CachedPortfolio struct {
    *Portfolio
    cachedValue  *decimal.Decimal
    cacheValid   bool
    lastModified time.Time
}

func (c *CachedPortfolio) GetValue() decimal.Decimal {
    if c.cacheValid && c.cachedValue != nil {
        return *c.cachedValue
    }

    // Recalculate value
    value := c.Portfolio.GetValue()
    c.cachedValue = &value
    c.cacheValid = true
    c.lastModified = time.Now()

    return value
}

func (c *CachedPortfolio) Add(component FinancialComponent) error {
    err := c.Portfolio.Add(component)
    if err == nil {
        c.invalidateCache()
    }
    return err
}

func (c *CachedPortfolio) Remove(component FinancialComponent) error {
    err := c.Portfolio.Remove(component)
    if err == nil {
        c.invalidateCache()
    }
    return err
}

func (c *CachedPortfolio) invalidateCache() {
    c.cacheValid = false
    c.lastModified = time.Now()
}
```

**2. Hierarchical Cache Invalidation**

```go
type CacheInvalidator interface {
    InvalidateCache()
}

type HierarchicalCachedPortfolio struct {
    *CachedPortfolio
    parent CacheInvalidator
}

func (h *HierarchicalCachedPortfolio) invalidateCache() {
    h.CachedPortfolio.invalidateCache()

    // Propagate invalidation up the hierarchy
    if h.parent != nil {
        h.parent.InvalidateCache()
    }
}

func (h *HierarchicalCachedPortfolio) Add(component FinancialComponent) error {
    err := h.CachedPortfolio.Add(component)
    if err == nil {
        // Set parent reference for cache invalidation
        if cached, ok := component.(*HierarchicalCachedPortfolio); ok {
            cached.parent = h
        }
    }
    return err
}
```

**3. Time-based Cache Expiration**

```go
type TimedCachedPortfolio struct {
    *Portfolio
    cachedValue    *decimal.Decimal
    cacheTimestamp time.Time
    cacheTTL       time.Duration
}

func (t *TimedCachedPortfolio) GetValue() decimal.Decimal {
    now := time.Now()

    if t.cachedValue != nil && now.Sub(t.cacheTimestamp) < t.cacheTTL {
        return *t.cachedValue
    }

    // Cache expired or invalid, recalculate
    value := t.Portfolio.GetValue()
    t.cachedValue = &value
    t.cacheTimestamp = now

    return value
}
```
