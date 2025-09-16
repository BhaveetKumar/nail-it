# Query Optimization

## Overview

This module covers query optimization concepts including query planning, cost estimation, join algorithms, and execution strategies. These concepts are essential for improving database performance and efficiency.

## Table of Contents

1. [Query Planning](#query-planning/)
2. [Cost Estimation](#cost-estimation/)
3. [Join Algorithms](#join-algorithms/)
4. [Index Selection](#index-selection/)
5. [Query Rewriting](#query-rewriting/)
6. [Applications](#applications/)
7. [Complexity Analysis](#complexity-analysis/)
8. [Follow-up Questions](#follow-up-questions/)

## Query Planning

### Theory

Query planning is the process of determining the most efficient way to execute a database query. It involves analyzing the query, considering different execution plans, and selecting the optimal one based on cost estimates.

### Query Planner Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "math"
    "sort"
    "strings"
)

type QueryType int

const (
    Select QueryType = iota
    Insert
    Update
    Delete
    Join
)

type Table struct {
    Name        string
    RowCount    int
    ColumnCount int
    Indexes     []string
    Size        int64
}

type Query struct {
    Type        QueryType
    Tables      []string
    Columns     []string
    Conditions  []Condition
    Joins       []Join
    OrderBy     []string
    GroupBy     []string
    Limit       int
    Offset      int
}

type Condition struct {
    Column   string
    Operator string
    Value    interface{}
}

type Join struct {
    Table1    string
    Table2    string
    Column1   string
    Column2   string
    Type      string // INNER, LEFT, RIGHT, FULL
}

type ExecutionPlan struct {
    Steps      []ExecutionStep
    TotalCost  float64
    EstimatedRows int
}

type ExecutionStep struct {
    Operation string
    Table     string
    Cost      float64
    Rows      int
    Index     string
}

type QueryOptimizer struct {
    tables map[string]*Table
    plans  []ExecutionPlan
}

func NewQueryOptimizer() *QueryOptimizer {
    return &QueryOptimizer{
        tables: make(map[string]*Table),
        plans:  make([]ExecutionPlan, 0),
    }
}

func (qo *QueryOptimizer) AddTable(name string, rowCount, columnCount int, size int64, indexes []string) {
    table := &Table{
        Name:        name,
        RowCount:    rowCount,
        ColumnCount: columnCount,
        Indexes:     indexes,
        Size:        size,
    }
    
    qo.tables[name] = table
    fmt.Printf("Added table: %s (%d rows, %d columns, %d bytes)\n", name, rowCount, columnCount, size)
}

func (qo *QueryOptimizer) OptimizeQuery(query *Query) *ExecutionPlan {
    fmt.Printf("Optimizing query on tables: %v\n", query.Tables)
    
    // Generate different execution plans
    plans := qo.generateExecutionPlans(query)
    
    // Select the best plan based on cost
    bestPlan := qo.selectBestPlan(plans)
    
    qo.plans = append(qo.plans, *bestPlan)
    return bestPlan
}

func (qo *QueryOptimizer) generateExecutionPlans(query *Query) []ExecutionPlan {
    var plans []ExecutionPlan
    
    // Plan 1: Sequential scan
    plan1 := qo.createSequentialScanPlan(query)
    plans = append(plans, plan1)
    
    // Plan 2: Index scan
    plan2 := qo.createIndexScanPlan(query)
    plans = append(plans, plan2)
    
    // Plan 3: Hash join
    if len(query.Joins) > 0 {
        plan3 := qo.createHashJoinPlan(query)
        plans = append(plans, plan3)
    }
    
    // Plan 4: Nested loop join
    if len(query.Joins) > 0 {
        plan4 := qo.createNestedLoopJoinPlan(query)
        plans = append(plans, plan4)
    }
    
    return plans
}

func (qo *QueryOptimizer) createSequentialScanPlan(query *Query) ExecutionPlan {
    var steps []ExecutionStep
    totalCost := 0.0
    estimatedRows := 0
    
    for _, tableName := range query.Tables {
        if table, exists := qo.tables[tableName]; exists {
            // Sequential scan cost
            cost := float64(table.RowCount) * 0.1 // Base cost per row
            
            step := ExecutionStep{
                Operation: "Sequential Scan",
                Table:     tableName,
                Cost:      cost,
                Rows:      table.RowCount,
                Index:     "",
            }
            
            steps = append(steps, step)
            totalCost += cost
            estimatedRows = table.RowCount
        }
    }
    
    return ExecutionPlan{
        Steps:        steps,
        TotalCost:    totalCost,
        EstimatedRows: estimatedRows,
    }
}

func (qo *QueryOptimizer) createIndexScanPlan(query *Query) ExecutionPlan {
    var steps []ExecutionStep
    totalCost := 0.0
    estimatedRows := 0
    
    for _, tableName := range query.Tables {
        if table, exists := qo.tables[tableName]; exists {
            // Check if there's an index for the conditions
            indexUsed := ""
            cost := float64(table.RowCount) * 0.1 // Default to sequential scan
            
            for _, condition := range query.Conditions {
                for _, index := range table.Indexes {
                    if strings.Contains(index, condition.Column) {
                        // Index scan cost
                        cost = math.Log2(float64(table.RowCount)) * 0.5
                        indexUsed = index
                        break
                    }
                }
            }
            
            step := ExecutionStep{
                Operation: "Index Scan",
                Table:     tableName,
                Cost:      cost,
                Rows:      table.RowCount,
                Index:     indexUsed,
            }
            
            steps = append(steps, step)
            totalCost += cost
            estimatedRows = table.RowCount
        }
    }
    
    return ExecutionPlan{
        Steps:        steps,
        TotalCost:    totalCost,
        EstimatedRows: estimatedRows,
    }
}

func (qo *QueryOptimizer) createHashJoinPlan(query *Query) ExecutionPlan {
    var steps []ExecutionStep
    totalCost := 0.0
    estimatedRows := 0
    
    // Build hash table for first table
    if len(query.Tables) > 0 {
        table1 := query.Tables[0]
        if table, exists := qo.tables[table1]; exists {
            cost := float64(table.RowCount) * 0.2 // Hash table build cost
            
            step := ExecutionStep{
                Operation: "Hash Build",
                Table:     table1,
                Cost:      cost,
                Rows:      table.RowCount,
                Index:     "",
            }
            
            steps = append(steps, step)
            totalCost += cost
            estimatedRows = table.RowCount
        }
    }
    
    // Probe hash table with second table
    if len(query.Tables) > 1 {
        table2 := query.Tables[1]
        if table, exists := qo.tables[table2]; exists {
            cost := float64(table.RowCount) * 0.1 // Hash probe cost
            
            step := ExecutionStep{
                Operation: "Hash Probe",
                Table:     table2,
                Cost:      cost,
                Rows:      table.RowCount,
                Index:     "",
            }
            
            steps = append(steps, step)
            totalCost += cost
            estimatedRows = int(float64(estimatedRows) * float64(table.RowCount) * 0.1) // Join selectivity
        }
    }
    
    return ExecutionPlan{
        Steps:        steps,
        TotalCost:    totalCost,
        EstimatedRows: estimatedRows,
    }
}

func (qo *QueryOptimizer) createNestedLoopJoinPlan(query *Query) ExecutionPlan {
    var steps []ExecutionStep
    totalCost := 0.0
    estimatedRows := 0
    
    if len(query.Tables) >= 2 {
        table1 := query.Tables[0]
        table2 := query.Tables[1]
        
        if t1, exists1 := qo.tables[table1]; exists1 {
            if t2, exists2 := qo.tables[table2]; exists2 {
                // Nested loop join cost
                cost := float64(t1.RowCount) * float64(t2.RowCount) * 0.01
                
                step := ExecutionStep{
                    Operation: "Nested Loop Join",
                    Table:     fmt.Sprintf("%s JOIN %s", table1, table2),
                    Cost:      cost,
                    Rows:      t1.RowCount * t2.RowCount,
                    Index:     "",
                }
                
                steps = append(steps, step)
                totalCost += cost
                estimatedRows = t1.RowCount * t2.RowCount
            }
        }
    }
    
    return ExecutionPlan{
        Steps:        steps,
        TotalCost:    totalCost,
        EstimatedRows: estimatedRows,
    }
}

func (qo *QueryOptimizer) selectBestPlan(plans []ExecutionPlan) *ExecutionPlan {
    if len(plans) == 0 {
        return nil
    }
    
    bestPlan := &plans[0]
    
    for i := 1; i < len(plans); i++ {
        if plans[i].TotalCost < bestPlan.TotalCost {
            bestPlan = &plans[i]
        }
    }
    
    return bestPlan
}

func (qo *QueryOptimizer) PrintExecutionPlan(plan *ExecutionPlan) {
    if plan == nil {
        fmt.Println("No execution plan available")
        return
    }
    
    fmt.Println("Execution Plan:")
    fmt.Printf("Total Cost: %.2f\n", plan.TotalCost)
    fmt.Printf("Estimated Rows: %d\n", plan.EstimatedRows)
    fmt.Println("Steps:")
    
    for i, step := range plan.Steps {
        fmt.Printf("  %d. %s on %s (Cost: %.2f, Rows: %d", i+1, step.Operation, step.Table, step.Cost, step.Rows)
        if step.Index != "" {
            fmt.Printf(", Index: %s", step.Index)
        }
        fmt.Println(")")
    }
}

func (qo *QueryOptimizer) GetTableStats(tableName string) {
    if table, exists := qo.tables[tableName]; exists {
        fmt.Printf("Table %s Statistics:\n", tableName)
        fmt.Printf("  Rows: %d\n", table.RowCount)
        fmt.Printf("  Columns: %d\n", table.ColumnCount)
        fmt.Printf("  Size: %d bytes\n", table.Size)
        fmt.Printf("  Indexes: %v\n", table.Indexes)
    } else {
        fmt.Printf("Table %s not found\n", tableName)
    }
}

func main() {
    optimizer := NewQueryOptimizer()
    
    fmt.Println("Query Optimization Demo:")
    
    // Add some tables
    optimizer.AddTable("Student", 10000, 5, 1000000, []string{"idx_student_id", "idx_student_name"})
    optimizer.AddTable("Course", 1000, 4, 100000, []string{"idx_course_id"})
    optimizer.AddTable("Enrollment", 50000, 3, 500000, []string{"idx_enrollment_student", "idx_enrollment_course"})
    
    // Create a sample query
    query := &Query{
        Type: Select,
        Tables: []string{"Student", "Enrollment"},
        Columns: []string{"Student.Name", "Enrollment.Grade"},
        Conditions: []Condition{
            {Column: "Student.StudentID", Operator: "=", Value: 123},
        },
        Joins: []Join{
            {Table1: "Student", Table2: "Enrollment", Column1: "StudentID", Column2: "StudentID", Type: "INNER"},
        },
    }
    
    // Optimize the query
    plan := optimizer.OptimizeQuery(query)
    
    // Print the execution plan
    optimizer.PrintExecutionPlan(plan)
    
    // Print table statistics
    fmt.Println("\nTable Statistics:")
    optimizer.GetTableStats("Student")
    optimizer.GetTableStats("Enrollment")
}
```

## Cost Estimation

### Theory

Cost estimation is the process of predicting the computational cost of executing a query plan. It involves estimating the number of rows processed, I/O operations, and CPU usage for each operation.

### Cost Estimator Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "math"
)

type CostModel struct {
    // Cost constants
    CPU_COST_PER_ROW    float64
    IO_COST_PER_PAGE    float64
    MEMORY_COST_PER_ROW float64
    PAGE_SIZE           int
}

type Statistics struct {
    TableName     string
    RowCount      int
    PageCount     int
    DistinctValues map[string]int
    MinValues     map[string]interface{}
    MaxValues     map[string]interface{}
    NullCounts    map[string]int
}

type CostEstimator struct {
    costModel   CostModel
    statistics  map[string]*Statistics
}

func NewCostEstimator() *CostEstimator {
    return &CostEstimator{
        costModel: CostModel{
            CPU_COST_PER_ROW:    0.01,
            IO_COST_PER_PAGE:    1.0,
            MEMORY_COST_PER_ROW: 0.001,
            PAGE_SIZE:           4096,
        },
        statistics: make(map[string]*Statistics),
    }
}

func (ce *CostEstimator) AddStatistics(tableName string, stats *Statistics) {
    ce.statistics[tableName] = stats
    fmt.Printf("Added statistics for table: %s\n", tableName)
}

func (ce *CostEstimator) EstimateSelectivity(condition Condition, tableName string) float64 {
    stats, exists := ce.statistics[tableName]
    if !exists {
        return 0.1 // Default selectivity
    }
    
    switch condition.Operator {
    case "=":
        // Equality condition
        if distinctCount, exists := stats.DistinctValues[condition.Column]; exists {
            return 1.0 / float64(distinctCount)
        }
        return 0.1
        
    case ">", ">=", "<", "<=":
        // Range condition
        return 0.3 // Assume 30% selectivity for range conditions
        
    case "LIKE":
        // Pattern matching
        return 0.05 // Assume 5% selectivity for LIKE conditions
        
    case "IN":
        // IN clause
        if distinctCount, exists := stats.DistinctValues[condition.Column]; exists {
            return 0.1 / float64(distinctCount) // Assume 10% of distinct values
        }
        return 0.1
        
    default:
        return 0.1
    }
}

func (ce *CostEstimator) EstimateJoinSelectivity(join Join, table1Stats, table2Stats *Statistics) float64 {
    // Simple join selectivity estimation
    // In practice, this would be more sophisticated
    
    if table1Stats == nil || table2Stats == nil {
        return 0.1
    }
    
    // Assume join selectivity based on smaller table
    smallerTableRows := math.Min(float64(table1Stats.RowCount), float64(table2Stats.RowCount))
    largerTableRows := math.Max(float64(table1Stats.RowCount), float64(table2Stats.RowCount))
    
    return smallerTableRows / largerTableRows
}

func (ce *CostEstimator) EstimateSequentialScanCost(tableName string, conditions []Condition) float64 {
    stats, exists := ce.statistics[tableName]
    if !exists {
        return 1000.0 // Default cost
    }
    
    // Base cost for reading all pages
    ioCost := float64(stats.PageCount) * ce.costModel.IO_COST_PER_PAGE
    
    // CPU cost for processing all rows
    cpuCost := float64(stats.RowCount) * ce.costModel.CPU_COST_PER_ROW
    
    // Apply selectivity for conditions
    selectivity := 1.0
    for _, condition := range conditions {
        selectivity *= ce.EstimateSelectivity(condition, tableName)
    }
    
    totalCost := (ioCost + cpuCost) * selectivity
    return totalCost
}

func (ce *CostEstimator) EstimateIndexScanCost(tableName string, indexName string, conditions []Condition) float64 {
    stats, exists := ce.statistics[tableName]
    if !exists {
        return 1000.0 // Default cost
    }
    
    // Index scan cost
    indexPages := int(math.Ceil(float64(stats.RowCount) / 100.0)) // Assume 100 rows per index page
    ioCost := float64(indexPages) * ce.costModel.IO_COST_PER_PAGE
    
    // CPU cost for index operations
    cpuCost := math.Log2(float64(stats.RowCount)) * ce.costModel.CPU_COST_PER_ROW * 10
    
    // Apply selectivity
    selectivity := 1.0
    for _, condition := range conditions {
        selectivity *= ce.EstimateSelectivity(condition, tableName)
    }
    
    totalCost := (ioCost + cpuCost) * selectivity
    return totalCost
}

func (ce *CostEstimator) EstimateHashJoinCost(table1Name, table2Name string, join Join) float64 {
    stats1, exists1 := ce.statistics[table1Name]
    stats2, exists2 := ce.statistics[table2Name]
    
    if !exists1 || !exists2 {
        return 1000.0 // Default cost
    }
    
    // Build phase cost
    buildCost := float64(stats1.RowCount) * ce.costModel.CPU_COST_PER_ROW
    buildCost += float64(stats1.PageCount) * ce.costModel.IO_COST_PER_PAGE
    
    // Probe phase cost
    probeCost := float64(stats2.RowCount) * ce.costModel.CPU_COST_PER_ROW
    probeCost += float64(stats2.PageCount) * ce.costModel.IO_COST_PER_PAGE
    
    // Join selectivity
    selectivity := ce.EstimateJoinSelectivity(join, stats1, stats2)
    
    totalCost := (buildCost + probeCost) * selectivity
    return totalCost
}

func (ce *CostEstimator) EstimateNestedLoopJoinCost(table1Name, table2Name string, join Join) float64 {
    stats1, exists1 := ce.statistics[table1Name]
    stats2, exists2 := ce.statistics[table2Name]
    
    if !exists1 || !exists2 {
        return 1000.0 // Default cost
    }
    
    // Nested loop join cost
    cost := float64(stats1.RowCount) * float64(stats2.RowCount) * ce.costModel.CPU_COST_PER_ROW
    
    // Join selectivity
    selectivity := ce.EstimateJoinSelectivity(join, stats1, stats2)
    
    totalCost := cost * selectivity
    return totalCost
}

func (ce *CostEstimator) EstimateQueryCost(query *Query) float64 {
    totalCost := 0.0
    
    // Estimate cost for each table access
    for _, tableName := range query.Tables {
        if len(query.Conditions) > 0 {
            // Try both sequential scan and index scan
            seqCost := ce.EstimateSequentialScanCost(tableName, query.Conditions)
            idxCost := ce.EstimateIndexScanCost(tableName, "idx_"+tableName, query.Conditions)
            
            // Choose the cheaper option
            if idxCost < seqCost {
                totalCost += idxCost
            } else {
                totalCost += seqCost
            }
        } else {
            // No conditions, use sequential scan
            totalCost += ce.EstimateSequentialScanCost(tableName, []Condition{})
        }
    }
    
    // Estimate cost for joins
    for _, join := range query.Joins {
        hashCost := ce.EstimateHashJoinCost(join.Table1, join.Table2, join)
        nestedCost := ce.EstimateNestedLoopJoinCost(join.Table1, join.Table2, join)
        
        // Choose the cheaper join method
        if hashCost < nestedCost {
            totalCost += hashCost
        } else {
            totalCost += nestedCost
        }
    }
    
    return totalCost
}

func (ce *CostEstimator) PrintStatistics(tableName string) {
    if stats, exists := ce.statistics[tableName]; exists {
        fmt.Printf("Statistics for table %s:\n", tableName)
        fmt.Printf("  Row Count: %d\n", stats.RowCount)
        fmt.Printf("  Page Count: %d\n", stats.PageCount)
        fmt.Printf("  Distinct Values:\n")
        for column, count := range stats.DistinctValues {
            fmt.Printf("    %s: %d\n", column, count)
        }
    } else {
        fmt.Printf("No statistics available for table %s\n", tableName)
    }
}

func main() {
    estimator := NewCostEstimator()
    
    fmt.Println("Cost Estimation Demo:")
    
    // Add statistics for tables
    studentStats := &Statistics{
        TableName: "Student",
        RowCount:  10000,
        PageCount: 100,
        DistinctValues: map[string]int{
            "StudentID": 10000,
            "Name":      9500,
            "Age":       50,
        },
        MinValues: map[string]interface{}{
            "StudentID": 1,
            "Age":       18,
        },
        MaxValues: map[string]interface{}{
            "StudentID": 10000,
            "Age":       25,
        },
        NullCounts: map[string]int{
            "Name": 0,
            "Age":  100,
        },
    }
    
    courseStats := &Statistics{
        TableName: "Course",
        RowCount:  1000,
        PageCount: 10,
        DistinctValues: map[string]int{
            "CourseID": 1000,
            "Title":    1000,
        },
    }
    
    estimator.AddStatistics("Student", studentStats)
    estimator.AddStatistics("Course", courseStats)
    
    // Create a sample query
    query := &Query{
        Type: Select,
        Tables: []string{"Student", "Course"},
        Conditions: []Condition{
            {Column: "Student.Age", Operator: ">", Value: 20},
            {Column: "Course.CourseID", Operator: "=", Value: 101},
        },
        Joins: []Join{
            {Table1: "Student", Table2: "Course", Column1: "StudentID", Column2: "StudentID", Type: "INNER"},
        },
    }
    
    // Estimate query cost
    cost := estimator.EstimateQueryCost(query)
    fmt.Printf("Estimated query cost: %.2f\n", cost)
    
    // Print statistics
    fmt.Println("\nTable Statistics:")
    estimator.PrintStatistics("Student")
    estimator.PrintStatistics("Course")
}
```

## Join Algorithms

### Theory

Join algorithms determine how to combine rows from two or more tables. Common algorithms include nested loop join, hash join, and sort-merge join, each with different performance characteristics.

### Join Algorithm Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sort"
    "time"
)

type Row struct {
    ID    int
    Data  map[string]interface{}
}

type JoinResult struct {
    LeftRow  Row
    RightRow Row
}

type JoinAlgorithm struct {
    leftTable  []Row
    rightTable []Row
    leftKey    string
    rightKey   string
}

func NewJoinAlgorithm(leftTable, rightTable []Row, leftKey, rightKey string) *JoinAlgorithm {
    return &JoinAlgorithm{
        leftTable:  leftTable,
        rightTable: rightTable,
        leftKey:    leftKey,
        rightKey:   rightKey,
    }
}

func (ja *JoinAlgorithm) NestedLoopJoin() []JoinResult {
    start := time.Now()
    var results []JoinResult
    
    for _, leftRow := range ja.leftTable {
        for _, rightRow := range ja.rightTable {
            if ja.compareKeys(leftRow, rightRow) == 0 {
                results = append(results, JoinResult{
                    LeftRow:  leftRow,
                    RightRow: rightRow,
                })
            }
        }
    }
    
    duration := time.Since(start)
    fmt.Printf("Nested Loop Join: %d results in %v\n", len(results), duration)
    return results
}

func (ja *JoinAlgorithm) HashJoin() []JoinResult {
    start := time.Now()
    var results []JoinResult
    
    // Build hash table from right table
    hashTable := make(map[interface{}][]Row)
    for _, rightRow := range ja.rightTable {
        key := rightRow.Data[ja.rightKey]
        hashTable[key] = append(hashTable[key], rightRow)
    }
    
    // Probe hash table with left table
    for _, leftRow := range ja.leftTable {
        key := leftRow.Data[ja.leftKey]
        if matchingRows, exists := hashTable[key]; exists {
            for _, rightRow := range matchingRows {
                results = append(results, JoinResult{
                    LeftRow:  leftRow,
                    RightRow: rightRow,
                })
            }
        }
    }
    
    duration := time.Since(start)
    fmt.Printf("Hash Join: %d results in %v\n", len(results), duration)
    return results
}

func (ja *JoinAlgorithm) SortMergeJoin() []JoinResult {
    start := time.Now()
    var results []JoinResult
    
    // Sort both tables by join keys
    leftSorted := make([]Row, len(ja.leftTable))
    copy(leftSorted, ja.leftTable)
    sort.Slice(leftSorted, func(i, j int) bool {
        return ja.compareKeys(leftSorted[i], leftSorted[j]) < 0
    })
    
    rightSorted := make([]Row, len(ja.rightTable))
    copy(rightSorted, ja.rightTable)
    sort.Slice(rightSorted, func(i, j int) bool {
        return ja.compareKeys(rightSorted[i], rightSorted[j]) < 0
    })
    
    // Merge the sorted tables
    leftIdx, rightIdx := 0, 0
    
    for leftIdx < len(leftSorted) && rightIdx < len(rightSorted) {
        cmp := ja.compareKeys(leftSorted[leftIdx], rightSorted[rightIdx])
        
        if cmp < 0 {
            leftIdx++
        } else if cmp > 0 {
            rightIdx++
        } else {
            // Keys match, find all matching rows
            leftStart := leftIdx
            rightStart := rightIdx
            
            // Find all left rows with same key
            for leftIdx < len(leftSorted) && ja.compareKeys(leftSorted[leftIdx], rightSorted[rightStart]) == 0 {
                leftIdx++
            }
            
            // Find all right rows with same key
            for rightIdx < len(rightSorted) && ja.compareKeys(leftSorted[leftStart], rightSorted[rightIdx]) == 0 {
                rightIdx++
            }
            
            // Cross product of matching rows
            for i := leftStart; i < leftIdx; i++ {
                for j := rightStart; j < rightIdx; j++ {
                    results = append(results, JoinResult{
                        LeftRow:  leftSorted[i],
                        RightRow: rightSorted[j],
                    })
                }
            }
        }
    }
    
    duration := time.Since(start)
    fmt.Printf("Sort-Merge Join: %d results in %v\n", len(results), duration)
    return results
}

func (ja *JoinAlgorithm) compareKeys(leftRow, rightRow Row) int {
    leftKey := leftRow.Data[ja.leftKey]
    rightKey := rightRow.Data[ja.rightKey]
    
    switch leftVal := leftKey.(type) {
    case int:
        rightVal := rightKey.(int)
        if leftVal < rightVal {
            return -1
        } else if leftVal > rightVal {
            return 1
        }
        return 0
    case string:
        rightVal := rightKey.(string)
        if leftVal < rightVal {
            return -1
        } else if leftVal > rightVal {
            return 1
        }
        return 0
    default:
        return 0
    }
}

func (ja *JoinAlgorithm) BenchmarkJoins() {
    fmt.Println("Join Algorithm Benchmark:")
    fmt.Printf("Left table size: %d rows\n", len(ja.leftTable))
    fmt.Printf("Right table size: %d rows\n", len(ja.rightTable))
    fmt.Println()
    
    // Benchmark nested loop join
    results1 := ja.NestedLoopJoin()
    
    // Benchmark hash join
    results2 := ja.HashJoin()
    
    // Benchmark sort-merge join
    results3 := ja.SortMergeJoin()
    
    // Verify results are the same
    if len(results1) == len(results2) && len(results2) == len(results3) {
        fmt.Println("All join algorithms produced the same number of results")
    } else {
        fmt.Printf("Result count mismatch: Nested=%d, Hash=%d, Sort-Merge=%d\n", 
                   len(results1), len(results2), len(results3))
    }
}

func main() {
    // Create sample data
    leftTable := make([]Row, 1000)
    rightTable := make([]Row, 1000)
    
    for i := 0; i < 1000; i++ {
        leftTable[i] = Row{
            ID: i,
            Data: map[string]interface{}{
                "ID":   i,
                "Name": fmt.Sprintf("Left_%d", i),
            },
        }
        
        rightTable[i] = Row{
            ID: i,
            Data: map[string]interface{}{
                "ID":   i,
                "Name": fmt.Sprintf("Right_%d", i),
            },
        }
    }
    
    // Create join algorithm
    ja := NewJoinAlgorithm(leftTable, rightTable, "ID", "ID")
    
    // Benchmark join algorithms
    ja.BenchmarkJoins()
}
```

## Follow-up Questions

### 1. Query Planning
**Q: What factors should be considered when choosing between different join algorithms?**
A: Consider table sizes, available memory, data distribution, index availability, and query selectivity. Hash joins are good for large tables with sufficient memory, nested loop joins for small tables, and sort-merge joins for pre-sorted data.

### 2. Cost Estimation
**Q: How can you improve the accuracy of cost estimation?**
A: Use up-to-date statistics, consider data distribution and correlation, use histograms for range queries, and continuously update statistics based on actual query performance.

### 3. Index Selection
**Q: When should you create composite indexes versus single-column indexes?**
A: Create composite indexes when queries frequently filter or sort by multiple columns together. Use single-column indexes for queries that only filter by one column or when the columns are used independently.

## Complexity Analysis

| Algorithm | Time Complexity | Space Complexity | Best Use Case |
|-----------|----------------|------------------|---------------|
| Nested Loop Join | O(n*m) | O(1) | Small tables |
| Hash Join | O(n+m) | O(min(n,m)) | Large tables with memory |
| Sort-Merge Join | O(n log n + m log m) | O(1) | Pre-sorted data |
| Sequential Scan | O(n) | O(1) | No indexes available |
| Index Scan | O(log n + k) | O(1) | Indexed columns |

## Applications

1. **Query Planning**: Database systems, query optimizers
2. **Cost Estimation**: Query optimizers, performance tuning
3. **Join Algorithms**: Database engines, data processing systems
4. **Index Selection**: Database administration, performance optimization

---

**Next**: [Transaction Management](transaction-management.md/) | **Previous**: [Database Systems](README.md/) | **Up**: [Database Systems](README.md/)
