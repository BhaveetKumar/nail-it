# Discrete Mathematics for Engineers

## Table of Contents

1. [Overview](#overview)
2. [Logic and Proofs](#logic-and-proofs)
3. [Set Theory](#set-theory)
4. [Graph Theory](#graph-theory)
5. [Combinatorics](#combinatorics)
6. [Number Theory](#number-theory)
7. [Applications](#applications)
8. [Implementations](#implementations)
9. [Follow-up Questions](#follow-up-questions)
10. [Sources](#sources)
11. [Projects](#projects)

## Overview

### Learning Objectives

- Master logical reasoning and proof techniques
- Understand set theory and relations
- Learn graph theory fundamentals
- Apply combinatorics to counting problems
- Use number theory in cryptography and algorithms
- Apply discrete math to computer science problems

### What is Discrete Mathematics?

Discrete Mathematics deals with mathematical structures that are fundamentally discrete rather than continuous. It includes logic, set theory, graph theory, combinatorics, and number theory - all essential for computer science and algorithm design.

## Logic and Proofs

### 1. Propositional Logic

#### Logical Operations and Truth Tables

```go
package main

import "fmt"

type Proposition struct {
    Value bool
    Name  string
}

func NewProposition(name string, value bool) *Proposition {
    return &Proposition{
        Name:  name,
        Value: value,
    }
}

type LogicCalculator struct{}

func NewLogicCalculator() *LogicCalculator {
    return &LogicCalculator{}
}

// Logical operations
func (lc *LogicCalculator) Not(p *Proposition) *Proposition {
    return &Proposition{
        Name:  "¬" + p.Name,
        Value: !p.Value,
    }
}

func (lc *LogicCalculator) And(p, q *Proposition) *Proposition {
    return &Proposition{
        Name:  "(" + p.Name + " ∧ " + q.Name + ")",
        Value: p.Value && q.Value,
    }
}

func (lc *LogicCalculator) Or(p, q *Proposition) *Proposition {
    return &Proposition{
        Name:  "(" + p.Name + " ∨ " + q.Name + ")",
        Value: p.Value || q.Value,
    }
}

func (lc *LogicCalculator) Implies(p, q *Proposition) *Proposition {
    return &Proposition{
        Name:  "(" + p.Name + " → " + q.Name + ")",
        Value: !p.Value || q.Value,
    }
}

func (lc *LogicCalculator) Iff(p, q *Proposition) *Proposition {
    return &Proposition{
        Name:  "(" + p.Name + " ↔ " + q.Name + ")",
        Value: p.Value == q.Value,
    }
}

func (lc *LogicCalculator) Xor(p, q *Proposition) *Proposition {
    return &Proposition{
        Name:  "(" + p.Name + " ⊕ " + q.Name + ")",
        Value: p.Value != q.Value,
    }
}

// Truth table generator
func (lc *LogicCalculator) GenerateTruthTable(variables []string, expression func([]bool) bool) {
    n := len(variables)
    fmt.Printf("%-10s", "Row")
    for _, v := range variables {
        fmt.Printf("%-10s", v)
    }
    fmt.Printf("%-10s\n", "Result")
    fmt.Println("------------------------------------------------")
    
    for i := 0; i < (1 << n); i++ {
        values := make([]bool, n)
        for j := 0; j < n; j++ {
            values[j] = (i>>j)&1 == 1
        }
        
        fmt.Printf("%-10d", i+1)
        for _, v := range values {
            fmt.Printf("%-10t", v)
        }
        fmt.Printf("%-10t\n", expression(values))
    }
}

// Logical equivalences
func (lc *LogicCalculator) DeMorganAnd(p, q *Proposition) *Proposition {
    // ¬(p ∧ q) ≡ (¬p ∨ ¬q)
    notP := lc.Not(p)
    notQ := lc.Not(q)
    return lc.Or(notP, notQ)
}

func (lc *LogicCalculator) DeMorganOr(p, q *Proposition) *Proposition {
    // ¬(p ∨ q) ≡ (¬p ∧ ¬q)
    notP := lc.Not(p)
    notQ := lc.Not(q)
    return lc.And(notP, notQ)
}

func (lc *LogicCalculator) DistributiveAnd(p, q, r *Proposition) *Proposition {
    // p ∧ (q ∨ r) ≡ (p ∧ q) ∨ (p ∧ r)
    qOrR := lc.Or(q, r)
    pAndQ := lc.And(p, q)
    pAndR := lc.And(p, r)
    return lc.Or(pAndQ, pAndR)
}

func (lc *LogicCalculator) DistributiveOr(p, q, r *Proposition) *Proposition {
    // p ∨ (q ∧ r) ≡ (p ∨ q) ∧ (p ∨ r)
    qAndR := lc.And(q, r)
    pOrQ := lc.Or(p, q)
    pOrR := lc.Or(p, r)
    return lc.And(pOrQ, pOrR)
}

func main() {
    calc := NewLogicCalculator()
    
    // Example propositions
    p := NewProposition("p", true)
    q := NewProposition("q", false)
    
    fmt.Println("Proposition p:", p.Value)
    fmt.Println("Proposition q:", q.Value)
    
    // Logical operations
    fmt.Println("¬p:", calc.Not(p).Value)
    fmt.Println("p ∧ q:", calc.And(p, q).Value)
    fmt.Println("p ∨ q:", calc.Or(p, q).Value)
    fmt.Println("p → q:", calc.Implies(p, q).Value)
    fmt.Println("p ↔ q:", calc.Iff(p, q).Value)
    fmt.Println("p ⊕ q:", calc.Xor(p, q).Value)
    
    // Truth table for p ∧ q
    fmt.Println("\nTruth table for p ∧ q:")
    calc.GenerateTruthTable([]string{"p", "q"}, func(values []bool) bool {
        return values[0] && values[1]
    })
    
    // De Morgan's laws
    fmt.Println("\nDe Morgan's Laws:")
    fmt.Println("¬(p ∧ q) ≡ (¬p ∨ ¬q):", calc.DeMorganAnd(p, q).Value)
    fmt.Println("¬(p ∨ q) ≡ (¬p ∧ ¬q):", calc.DeMorganOr(p, q).Value)
}
```

#### Node.js Implementation

```javascript
class Proposition {
    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
}

class LogicCalculator {
    not(p) {
        return new Proposition(`¬${p.name}`, !p.value);
    }
    
    and(p, q) {
        return new Proposition(`(${p.name} ∧ ${q.name})`, p.value && q.value);
    }
    
    or(p, q) {
        return new Proposition(`(${p.name} ∨ ${q.name})`, p.value || q.value);
    }
    
    implies(p, q) {
        return new Proposition(`(${p.name} → ${q.name})`, !p.value || q.value);
    }
    
    iff(p, q) {
        return new Proposition(`(${p.name} ↔ ${q.name})`, p.value === q.value);
    }
    
    xor(p, q) {
        return new Proposition(`(${p.name} ⊕ ${q.name})`, p.value !== q.value);
    }
    
    generateTruthTable(variables, expression) {
        const n = variables.length;
        console.log(`${'Row'.padEnd(10)}${variables.map(v => v.padEnd(10)).join('')}Result`);
        console.log('-'.repeat(50));
        
        for (let i = 0; i < (1 << n); i++) {
            const values = [];
            for (let j = 0; j < n; j++) {
                values.push((i >> j) & 1 === 1);
            }
            
            const row = `${(i + 1).toString().padEnd(10)}`;
            const valuesStr = values.map(v => v.toString().padEnd(10)).join('');
            const result = expression(values).toString().padEnd(10);
            console.log(row + valuesStr + result);
        }
    }
    
    deMorganAnd(p, q) {
        const notP = this.not(p);
        const notQ = this.not(q);
        return this.or(notP, notQ);
    }
    
    deMorganOr(p, q) {
        const notP = this.not(p);
        const notQ = this.not(q);
        return this.and(notP, notQ);
    }
}

// Example usage
const calc = new LogicCalculator();
const p = new Proposition('p', true);
const q = new Proposition('q', false);

console.log('Proposition p:', p.value);
console.log('Proposition q:', q.value);
console.log('¬p:', calc.not(p).value);
console.log('p ∧ q:', calc.and(p, q).value);
console.log('p ∨ q:', calc.or(p, q).value);
console.log('p → q:', calc.implies(p, q).value);
console.log('p ↔ q:', calc.iff(p, q).value);
console.log('p ⊕ q:', calc.xor(p, q).value);

console.log('\nTruth table for p ∧ q:');
calc.generateTruthTable(['p', 'q'], (values) => values[0] && values[1]);
```

## Set Theory

### 1. Set Operations

#### Basic Set Operations

```go
package main

import (
    "fmt"
    "sort"
)

type Set struct {
    Elements map[interface{}]bool
}

func NewSet() *Set {
    return &Set{
        Elements: make(map[interface{}]bool),
    }
}

func NewSetFromSlice(slice []interface{}) *Set {
    s := NewSet()
    for _, element := range slice {
        s.Add(element)
    }
    return s
}

func (s *Set) Add(element interface{}) {
    s.Elements[element] = true
}

func (s *Set) Remove(element interface{}) {
    delete(s.Elements, element)
}

func (s *Set) Contains(element interface{}) bool {
    return s.Elements[element]
}

func (s *Set) Size() int {
    return len(s.Elements)
}

func (s *Set) IsEmpty() bool {
    return len(s.Elements) == 0
}

func (s *Set) Union(other *Set) *Set {
    result := NewSet()
    
    for element := range s.Elements {
        result.Add(element)
    }
    
    for element := range other.Elements {
        result.Add(element)
    }
    
    return result
}

func (s *Set) Intersection(other *Set) *Set {
    result := NewSet()
    
    for element := range s.Elements {
        if other.Contains(element) {
            result.Add(element)
        }
    }
    
    return result
}

func (s *Set) Difference(other *Set) *Set {
    result := NewSet()
    
    for element := range s.Elements {
        if !other.Contains(element) {
            result.Add(element)
        }
    }
    
    return result
}

func (s *Set) SymmetricDifference(other *Set) *Set {
    union := s.Union(other)
    intersection := s.Intersection(other)
    return union.Difference(intersection)
}

func (s *Set) IsSubset(other *Set) bool {
    for element := range s.Elements {
        if !other.Contains(element) {
            return false
        }
    }
    return true
}

func (s *Set) IsSuperset(other *Set) bool {
    return other.IsSubset(s)
}

func (s *Set) IsDisjoint(other *Set) bool {
    return s.Intersection(other).IsEmpty()
}

func (s *Set) PowerSet() []*Set {
    elements := s.ToSlice()
    n := len(elements)
    powerSet := make([]*Set, 0, 1<<n)
    
    for i := 0; i < (1 << n); i++ {
        subset := NewSet()
        for j := 0; j < n; j++ {
            if (i>>j)&1 == 1 {
                subset.Add(elements[j])
            }
        }
        powerSet = append(powerSet, subset)
    }
    
    return powerSet
}

func (s *Set) CartesianProduct(other *Set) [][]interface{} {
    result := make([][]interface{}, 0)
    
    for element1 := range s.Elements {
        for element2 := range other.Elements {
            result = append(result, []interface{}{element1, element2})
        }
    }
    
    return result
}

func (s *Set) ToSlice() []interface{} {
    slice := make([]interface{}, 0, len(s.Elements))
    for element := range s.Elements {
        slice = append(slice, element)
    }
    return slice
}

func (s *Set) String() string {
    elements := s.ToSlice()
    sort.Slice(elements, func(i, j int) bool {
        return fmt.Sprintf("%v", elements[i]) < fmt.Sprintf("%v", elements[j])
    })
    
    result := "{"
    for i, element := range elements {
        if i > 0 {
            result += ", "
        }
        result += fmt.Sprintf("%v", element)
    }
    result += "}"
    return result
}

func main() {
    // Create sets
    set1 := NewSetFromSlice([]interface{}{1, 2, 3, 4, 5})
    set2 := NewSetFromSlice([]interface{}{4, 5, 6, 7, 8})
    
    fmt.Println("Set 1:", set1)
    fmt.Println("Set 2:", set2)
    
    // Set operations
    fmt.Println("Union:", set1.Union(set2))
    fmt.Println("Intersection:", set1.Intersection(set2))
    fmt.Println("Difference (1-2):", set1.Difference(set2))
    fmt.Println("Difference (2-1):", set2.Difference(set1))
    fmt.Println("Symmetric Difference:", set1.SymmetricDifference(set2))
    
    // Set properties
    fmt.Println("Is set1 subset of set2:", set1.IsSubset(set2))
    fmt.Println("Is set1 superset of set2:", set1.IsSuperset(set2))
    fmt.Println("Are sets disjoint:", set1.IsDisjoint(set2))
    
    // Power set (for small sets)
    smallSet := NewSetFromSlice([]interface{}{1, 2, 3})
    powerSet := smallSet.PowerSet()
    fmt.Println("Power set of {1, 2, 3}:")
    for i, subset := range powerSet {
        fmt.Printf("  %d: %s\n", i, subset)
    }
    
    // Cartesian product
    setA := NewSetFromSlice([]interface{}{"a", "b"})
    setB := NewSetFromSlice([]interface{}{1, 2})
    cartesian := setA.CartesianProduct(setB)
    fmt.Println("Cartesian product of {a, b} × {1, 2}:")
    for _, pair := range cartesian {
        fmt.Printf("  (%v, %v)\n", pair[0], pair[1])
    }
}
```

## Graph Theory

### 1. Graph Representation and Algorithms

#### Adjacency List and Matrix

```go
package main

import (
    "fmt"
    "math"
)

type Graph struct {
    Vertices map[int][]int
    Directed bool
}

func NewGraph(directed bool) *Graph {
    return &Graph{
        Vertices: make(map[int][]int),
        Directed: directed,
    }
}

func (g *Graph) AddVertex(vertex int) {
    if g.Vertices[vertex] == nil {
        g.Vertices[vertex] = make([]int, 0)
    }
}

func (g *Graph) AddEdge(from, to int) {
    g.AddVertex(from)
    g.AddVertex(to)
    
    g.Vertices[from] = append(g.Vertices[from], to)
    
    if !g.Directed {
        g.Vertices[to] = append(g.Vertices[to], from)
    }
}

func (g *Graph) RemoveEdge(from, to int) {
    if g.Vertices[from] != nil {
        for i, neighbor := range g.Vertices[from] {
            if neighbor == to {
                g.Vertices[from] = append(g.Vertices[from][:i], g.Vertices[from][i+1:]...)
                break
            }
        }
    }
    
    if !g.Directed && g.Vertices[to] != nil {
        for i, neighbor := range g.Vertices[to] {
            if neighbor == from {
                g.Vertices[to] = append(g.Vertices[to][:i], g.Vertices[to][i+1:]...)
                break
            }
        }
    }
}

func (g *Graph) HasEdge(from, to int) bool {
    if g.Vertices[from] == nil {
        return false
    }
    
    for _, neighbor := range g.Vertices[from] {
        if neighbor == to {
            return true
        }
    }
    return false
}

func (g *Graph) GetNeighbors(vertex int) []int {
    if g.Vertices[vertex] == nil {
        return []int{}
    }
    return g.Vertices[vertex]
}

func (g *Graph) GetVertices() []int {
    vertices := make([]int, 0, len(g.Vertices))
    for vertex := range g.Vertices {
        vertices = append(vertices, vertex)
    }
    return vertices
}

func (g *Graph) GetEdgeCount() int {
    count := 0
    for _, neighbors := range g.Vertices {
        count += len(neighbors)
    }
    
    if !g.Directed {
        count /= 2
    }
    
    return count
}

// Depth-First Search
func (g *Graph) DFS(start int) []int {
    visited := make(map[int]bool)
    result := make([]int, 0)
    
    g.dfsHelper(start, visited, &result)
    return result
}

func (g *Graph) dfsHelper(vertex int, visited map[int]bool, result *[]int) {
    visited[vertex] = true
    *result = append(*result, vertex)
    
    for _, neighbor := range g.GetNeighbors(vertex) {
        if !visited[neighbor] {
            g.dfsHelper(neighbor, visited, result)
        }
    }
}

// Breadth-First Search
func (g *Graph) BFS(start int) []int {
    visited := make(map[int]bool)
    queue := []int{start}
    result := make([]int, 0)
    
    visited[start] = true
    
    for len(queue) > 0 {
        vertex := queue[0]
        queue = queue[1:]
        result = append(result, vertex)
        
        for _, neighbor := range g.GetNeighbors(vertex) {
            if !visited[neighbor] {
                visited[neighbor] = true
                queue = append(queue, neighbor)
            }
        }
    }
    
    return result
}

// Shortest Path (BFS for unweighted graphs)
func (g *Graph) ShortestPath(start, end int) ([]int, int) {
    if start == end {
        return []int{start}, 0
    }
    
    visited := make(map[int]bool)
    parent := make(map[int]int)
    queue := []int{start}
    
    visited[start] = true
    
    for len(queue) > 0 {
        vertex := queue[0]
        queue = queue[1:]
        
        for _, neighbor := range g.GetNeighbors(vertex) {
            if !visited[neighbor] {
                visited[neighbor] = true
                parent[neighbor] = vertex
                queue = append(queue, neighbor)
                
                if neighbor == end {
                    // Reconstruct path
                    path := []int{end}
                    current := end
                    
                    for parent[current] != start {
                        current = parent[current]
                        path = append([]int{current}, path...)
                    }
                    path = append([]int{start}, path...)
                    
                    return path, len(path) - 1
                }
            }
        }
    }
    
    return []int{}, -1 // No path found
}

// Cycle Detection
func (g *Graph) HasCycle() bool {
    visited := make(map[int]bool)
    recStack := make(map[int]bool)
    
    for vertex := range g.Vertices {
        if !visited[vertex] {
            if g.hasCycleHelper(vertex, visited, recStack) {
                return true
            }
        }
    }
    
    return false
}

func (g *Graph) hasCycleHelper(vertex int, visited, recStack map[int]bool) bool {
    visited[vertex] = true
    recStack[vertex] = true
    
    for _, neighbor := range g.GetNeighbors(vertex) {
        if !visited[neighbor] {
            if g.hasCycleHelper(neighbor, visited, recStack) {
                return true
            }
        } else if recStack[neighbor] {
            return true
        }
    }
    
    recStack[vertex] = false
    return false
}

// Topological Sort (for DAGs)
func (g *Graph) TopologicalSort() []int {
    if !g.Directed {
        return []int{} // Topological sort only for DAGs
    }
    
    visited := make(map[int]bool)
    stack := make([]int, 0)
    
    for vertex := range g.Vertices {
        if !visited[vertex] {
            g.topologicalSortHelper(vertex, visited, &stack)
        }
    }
    
    // Reverse the stack
    result := make([]int, len(stack))
    for i, j := 0, len(stack)-1; i < len(stack); i, j = i+1, j-1 {
        result[i] = stack[j]
    }
    
    return result
}

func (g *Graph) topologicalSortHelper(vertex int, visited map[int]bool, stack *[]int) {
    visited[vertex] = true
    
    for _, neighbor := range g.GetNeighbors(vertex) {
        if !visited[neighbor] {
            g.topologicalSortHelper(neighbor, visited, stack)
        }
    }
    
    *stack = append(*stack, vertex)
}

func main() {
    // Create a directed graph
    graph := NewGraph(true)
    
    // Add edges
    graph.AddEdge(1, 2)
    graph.AddEdge(1, 3)
    graph.AddEdge(2, 4)
    graph.AddEdge(3, 4)
    graph.AddEdge(4, 5)
    
    fmt.Println("Graph vertices:", graph.GetVertices())
    fmt.Println("Graph edges:", graph.GetEdgeCount())
    
    // DFS traversal
    fmt.Println("DFS from vertex 1:", graph.DFS(1))
    
    // BFS traversal
    fmt.Println("BFS from vertex 1:", graph.BFS(1))
    
    // Shortest path
    path, length := graph.ShortestPath(1, 5)
    fmt.Printf("Shortest path from 1 to 5: %v (length: %d)\n", path, length)
    
    // Cycle detection
    fmt.Println("Has cycle:", graph.HasCycle())
    
    // Topological sort
    fmt.Println("Topological sort:", graph.TopologicalSort())
}
```

## Combinatorics

### 1. Counting Principles

#### Permutations, Combinations, and Pigeonhole Principle

```go
package main

import (
    "fmt"
    "math"
)

type Combinatorics struct{}

func NewCombinatorics() *Combinatorics {
    return &Combinatorics{}
}

func (c *Combinatorics) Factorial(n int) int64 {
    if n < 0 {
        return 0
    }
    if n <= 1 {
        return 1
    }
    
    result := int64(1)
    for i := 2; i <= n; i++ {
        result *= int64(i)
    }
    return result
}

func (c *Combinatorics) Permutation(n, r int) int64 {
    if n < 0 || r < 0 || r > n {
        return 0
    }
    return c.Factorial(n) / c.Factorial(n-r)
}

func (c *Combinatorics) Combination(n, r int) int64 {
    if n < 0 || r < 0 || r > n {
        return 0
    }
    return c.Factorial(n) / (c.Factorial(r) * c.Factorial(n-r))
}

func (c *Combinatorics) PermutationWithRepetition(n, r int) int64 {
    if n < 0 || r < 0 {
        return 0
    }
    return int64(math.Pow(float64(n), float64(r)))
}

func (c *Combinatorics) CombinationWithRepetition(n, r int) int64 {
    if n < 0 || r < 0 {
        return 0
    }
    return c.Combination(n+r-1, r)
}

func (c *Combinatorics) StirlingNumberSecondKind(n, k int) int64 {
    if n == 0 && k == 0 {
        return 1
    }
    if n == 0 || k == 0 {
        return 0
    }
    if k > n {
        return 0
    }
    
    // S(n,k) = k*S(n-1,k) + S(n-1,k-1)
    dp := make([][]int64, n+1)
    for i := range dp {
        dp[i] = make([]int64, k+1)
    }
    
    dp[0][0] = 1
    
    for i := 1; i <= n; i++ {
        for j := 1; j <= k && j <= i; j++ {
            dp[i][j] = int64(j)*dp[i-1][j] + dp[i-1][j-1]
        }
    }
    
    return dp[n][k]
}

func (c *Combinatorics) BellNumber(n int) int64 {
    if n < 0 {
        return 0
    }
    if n <= 1 {
        return 1
    }
    
    sum := int64(0)
    for k := 0; k <= n; k++ {
        sum += c.StirlingNumberSecondKind(n, k)
    }
    return sum
}

func (c *Combinatorics) CatalanNumber(n int) int64 {
    if n < 0 {
        return 0
    }
    if n <= 1 {
        return 1
    }
    
    // C(n) = (2n)! / ((n+1)! * n!)
    return c.Factorial(2*n) / (c.Factorial(n+1) * c.Factorial(n))
}

func (c *Combinatorics) Fibonacci(n int) int64 {
    if n < 0 {
        return 0
    }
    if n <= 1 {
        return int64(n)
    }
    
    a, b := int64(0), int64(1)
    for i := 2; i <= n; i++ {
        a, b = b, a+b
    }
    return b
}

func (c *Combinatorics) PigeonholePrinciple(n, k int) bool {
    // If n items are put into k containers with n > k,
    // then at least one container must contain more than one item
    return n > k
}

func (c *Combinatorics) GeneratePermutations(elements []int) [][]int {
    if len(elements) == 0 {
        return [][]int{{}}
    }
    
    if len(elements) == 1 {
        return [][]int{elements}
    }
    
    var result [][]int
    
    for i, element := range elements {
        // Create a copy without the current element
        remaining := make([]int, 0, len(elements)-1)
        remaining = append(remaining, elements[:i]...)
        remaining = append(remaining, elements[i+1:]...)
        
        // Generate permutations of remaining elements
        subPermutations := c.GeneratePermutations(remaining)
        
        // Add current element to the beginning of each sub-permutation
        for _, perm := range subPermutations {
            newPerm := make([]int, 0, len(perm)+1)
            newPerm = append(newPerm, element)
            newPerm = append(newPerm, perm...)
            result = append(result, newPerm)
        }
    }
    
    return result
}

func (c *Combinatorics) GenerateCombinations(elements []int, r int) [][]int {
    if r == 0 {
        return [][]int{{}}
    }
    if r > len(elements) {
        return [][]int{}
    }
    
    var result [][]int
    
    for i := 0; i <= len(elements)-r; i++ {
        // Take the first element
        first := elements[i]
        
        // Generate combinations of remaining elements
        remaining := elements[i+1:]
        subCombinations := c.GenerateCombinations(remaining, r-1)
        
        // Add first element to each sub-combination
        for _, comb := range subCombinations {
            newComb := make([]int, 0, len(comb)+1)
            newComb = append(newComb, first)
            newComb = append(newComb, comb...)
            result = append(result, newComb)
        }
    }
    
    return result
}

func main() {
    comb := NewCombinatorics()
    
    // Basic counting
    fmt.Println("Factorial of 5:", comb.Factorial(5))
    fmt.Println("P(10,3):", comb.Permutation(10, 3))
    fmt.Println("C(10,3):", comb.Combination(10, 3))
    fmt.Println("P(5,3) with repetition:", comb.PermutationWithRepetition(5, 3))
    fmt.Println("C(5,3) with repetition:", comb.CombinationWithRepetition(5, 3))
    
    // Special numbers
    fmt.Println("Stirling S(4,2):", comb.StirlingNumberSecondKind(4, 2))
    fmt.Println("Bell number B(4):", comb.BellNumber(4))
    fmt.Println("Catalan number C(4):", comb.CatalanNumber(4))
    fmt.Println("Fibonacci F(10):", comb.Fibonacci(10))
    
    // Pigeonhole principle
    fmt.Println("Pigeonhole principle (10 items, 3 boxes):", comb.PigeonholePrinciple(10, 3))
    
    // Generate permutations and combinations
    elements := []int{1, 2, 3}
    fmt.Println("Permutations of [1,2,3]:", comb.GeneratePermutations(elements))
    fmt.Println("Combinations of [1,2,3] choose 2:", comb.GenerateCombinations(elements, 2))
}
```

## Number Theory

### 1. Prime Numbers and Modular Arithmetic

#### GCD, LCM, and Prime Factorization

```go
package main

import (
    "fmt"
    "math"
)

type NumberTheory struct{}

func NewNumberTheory() *NumberTheory {
    return &NumberTheory{}
}

func (nt *NumberTheory) GCD(a, b int) int {
    if b == 0 {
        return a
    }
    return nt.GCD(b, a%b)
}

func (nt *NumberTheory) LCM(a, b int) int {
    return a * b / nt.GCD(a, b)
}

func (nt *NumberTheory) ExtendedGCD(a, b int) (int, int, int) {
    if b == 0 {
        return a, 1, 0
    }
    
    gcd, x1, y1 := nt.ExtendedGCD(b, a%b)
    x := y1
    y := x1 - (a/b)*y1
    
    return gcd, x, y
}

func (nt *NumberTheory) IsPrime(n int) bool {
    if n < 2 {
        return false
    }
    if n == 2 {
        return true
    }
    if n%2 == 0 {
        return false
    }
    
    for i := 3; i*i <= n; i += 2 {
        if n%i == 0 {
            return false
        }
    }
    
    return true
}

func (nt *NumberTheory) SieveOfEratosthenes(n int) []int {
    if n < 2 {
        return []int{}
    }
    
    isPrime := make([]bool, n+1)
    for i := 2; i <= n; i++ {
        isPrime[i] = true
    }
    
    for i := 2; i*i <= n; i++ {
        if isPrime[i] {
            for j := i * i; j <= n; j += i {
                isPrime[j] = false
            }
        }
    }
    
    var primes []int
    for i := 2; i <= n; i++ {
        if isPrime[i] {
            primes = append(primes, i)
        }
    }
    
    return primes
}

func (nt *NumberTheory) PrimeFactors(n int) []int {
    if n < 2 {
        return []int{}
    }
    
    var factors []int
    
    // Check for 2
    for n%2 == 0 {
        factors = append(factors, 2)
        n /= 2
    }
    
    // Check for odd numbers
    for i := 3; i*i <= n; i += 2 {
        for n%i == 0 {
            factors = append(factors, i)
            n /= i
        }
    }
    
    // If n is still greater than 1, it's a prime factor
    if n > 1 {
        factors = append(factors, n)
    }
    
    return factors
}

func (nt *NumberTheory) Totient(n int) int {
    if n < 1 {
        return 0
    }
    
    result := n
    factors := nt.UniquePrimeFactors(n)
    
    for _, p := range factors {
        result = result * (p - 1) / p
    }
    
    return result
}

func (nt *NumberTheory) UniquePrimeFactors(n int) []int {
    factors := nt.PrimeFactors(n)
    unique := make([]int, 0)
    seen := make(map[int]bool)
    
    for _, factor := range factors {
        if !seen[factor] {
            unique = append(unique, factor)
            seen[factor] = true
        }
    }
    
    return unique
}

func (nt *NumberTheory) ModularExponentiation(base, exponent, modulus int) int {
    if modulus == 1 {
        return 0
    }
    
    result := 1
    base = base % modulus
    
    for exponent > 0 {
        if exponent%2 == 1 {
            result = (result * base) % modulus
        }
        exponent = exponent >> 1
        base = (base * base) % modulus
    }
    
    return result
}

func (nt *NumberTheory) ModularInverse(a, m int) int {
    gcd, x, _ := nt.ExtendedGCD(a, m)
    if gcd != 1 {
        return -1 // Inverse doesn't exist
    }
    
    return ((x % m) + m) % m
}

func (nt *NumberTheory) ChineseRemainderTheorem(remainders, moduli []int) int {
    if len(remainders) != len(moduli) {
        return -1
    }
    
    n := len(remainders)
    product := 1
    for _, m := range moduli {
        product *= m
    }
    
    result := 0
    for i := 0; i < n; i++ {
        mi := product / moduli[i]
        _, _, yi := nt.ExtendedGCD(mi, moduli[i])
        result += remainders[i] * mi * yi
    }
    
    return result % product
}

func (nt *NumberTheory) IsCarmichael(n int) bool {
    if n < 2 || nt.IsPrime(n) {
        return false
    }
    
    for a := 2; a < n; a++ {
        if nt.GCD(a, n) == 1 {
            if nt.ModularExponentiation(a, n-1, n) != 1 {
                return false
            }
        }
    }
    
    return true
}

func main() {
    nt := NewNumberTheory()
    
    // Basic operations
    fmt.Println("GCD(48, 18):", nt.GCD(48, 18))
    fmt.Println("LCM(48, 18):", nt.LCM(48, 18))
    
    gcd, x, y := nt.ExtendedGCD(48, 18)
    fmt.Printf("Extended GCD(48, 18): gcd=%d, x=%d, y=%d\n", gcd, x, y)
    
    // Prime numbers
    fmt.Println("Is 17 prime:", nt.IsPrime(17))
    fmt.Println("Is 15 prime:", nt.IsPrime(15))
    fmt.Println("Primes up to 30:", nt.SieveOfEratosthenes(30))
    
    // Prime factorization
    fmt.Println("Prime factors of 60:", nt.PrimeFactors(60))
    fmt.Println("Unique prime factors of 60:", nt.UniquePrimeFactors(60))
    fmt.Println("Euler's totient φ(10):", nt.Totient(10))
    
    // Modular arithmetic
    fmt.Println("2^10 mod 7:", nt.ModularExponentiation(2, 10, 7))
    fmt.Println("Modular inverse of 3 mod 7:", nt.ModularInverse(3, 7))
    
    // Chinese Remainder Theorem
    remainders := []int{2, 3, 2}
    moduli := []int{3, 5, 7}
    crt := nt.ChineseRemainderTheorem(remainders, moduli)
    fmt.Printf("Chinese Remainder Theorem: x ≡ %d (mod %d)\n", crt, 3*5*7)
    
    // Carmichael numbers
    fmt.Println("Is 561 a Carmichael number:", nt.IsCarmichael(561))
}
```

## Follow-up Questions

### 1. Logic and Proofs
**Q: What's the difference between a necessary and sufficient condition?**
A: A condition A is necessary for B if B cannot be true without A. A condition A is sufficient for B if A being true guarantees B is true.

### 2. Set Theory
**Q: What's the cardinality of the power set of a set with n elements?**
A: The cardinality of the power set is 2^n. This is because each element can either be included or excluded from a subset, giving 2 choices for each of the n elements.

### 3. Graph Theory
**Q: What's the difference between a path and a cycle?**
A: A path is a sequence of vertices where each consecutive pair is connected by an edge, and no vertex is repeated. A cycle is a path that starts and ends at the same vertex.

## Sources

### Books
- **Discrete Mathematics and Its Applications** by Kenneth Rosen
- **Concrete Mathematics** by Graham, Knuth, Patashnik
- **Introduction to Graph Theory** by Douglas West

### Online Resources
- **Khan Academy** - Discrete mathematics
- **Coursera** - Discrete mathematics courses
- **MIT OpenCourseWare** - Mathematics for computer science

## Projects

### 1. Logic Calculator
**Objective**: Build a propositional logic calculator
**Requirements**: Truth tables, logical equivalences, proof checking
**Deliverables**: Complete logic calculator application

### 2. Graph Algorithm Visualizer
**Objective**: Create a tool to visualize graph algorithms
**Requirements**: Graph representation, algorithm animations
**Deliverables**: Interactive graph algorithm visualizer

### 3. Combinatorics Library
**Objective**: Implement a comprehensive combinatorics library
**Requirements**: Permutations, combinations, special numbers
**Deliverables**: Complete combinatorics library with examples

---

**Next**: [Programming Fundamentals](../../../README.md) | **Previous**: [Statistics & Probability](statistics-probability.md) | **Up**: [Phase 0](README.md)



## Applications

<!-- AUTO-GENERATED ANCHOR: originally referenced as #applications -->

Placeholder content. Please replace with proper section.


## Implementations

<!-- AUTO-GENERATED ANCHOR: originally referenced as #implementations -->

Placeholder content. Please replace with proper section.
