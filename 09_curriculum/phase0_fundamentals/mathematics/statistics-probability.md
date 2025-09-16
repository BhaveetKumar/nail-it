# Statistics & Probability for Engineers

## Table of Contents

1. [Overview](#overview/)
2. [Descriptive Statistics](#descriptive-statistics/)
3. [Probability Theory](#probability-theory/)
4. [Probability Distributions](#probability-distributions/)
5. [Hypothesis Testing](#hypothesis-testing/)
6. [Regression Analysis](#regression-analysis/)
7. [Applications](#applications/)
8. [Implementations](#implementations/)
9. [Follow-up Questions](#follow-up-questions/)
10. [Sources](#sources/)
11. [Projects](#projects/)

## Overview

### Learning Objectives

- Master descriptive statistics and data analysis
- Understand probability theory and random variables
- Learn common probability distributions
- Apply hypothesis testing and statistical inference
- Use regression analysis for prediction
- Apply statistics to machine learning and data science

### What is Statistics & Probability?

Statistics and Probability are mathematical disciplines that deal with data analysis, uncertainty quantification, and decision making under uncertainty. They're essential for machine learning, data science, quality control, and engineering design.

## Descriptive Statistics

### 1. Central Tendency and Dispersion

#### Basic Statistical Measures

```go
package main

import (
    "fmt"
    "math"
    "sort"
)

type Statistics struct {
    Data []float64
    N    int
}

func NewStatistics(data []float64) *Statistics {
    return &Statistics{
        Data: data,
        N:    len(data),
    }
}

func (s *Statistics) Mean() float64 {
    if s.N == 0 {
        return 0
    }
    
    sum := 0.0
    for _, value := range s.Data {
        sum += value
    }
    return sum / float64(s.N)
}

func (s *Statistics) Median() float64 {
    if s.N == 0 {
        return 0
    }
    
    sorted := make([]float64, s.N)
    copy(sorted, s.Data)
    sort.Float64s(sorted)
    
    if s.N%2 == 0 {
        return (sorted[s.N/2-1] + sorted[s.N/2]) / 2.0
    }
    return sorted[s.N/2]
}

func (s *Statistics) Mode() []float64 {
    if s.N == 0 {
        return []float64{}
    }
    
    frequency := make(map[float64]int)
    for _, value := range s.Data {
        frequency[value]++
    }
    
    maxFreq := 0
    for _, freq := range frequency {
        if freq > maxFreq {
            maxFreq = freq
        }
    }
    
    var modes []float64
    for value, freq := range frequency {
        if freq == maxFreq {
            modes = append(modes, value)
        }
    }
    
    return modes
}

func (s *Statistics) Variance() float64 {
    if s.N <= 1 {
        return 0
    }
    
    mean := s.Mean()
    sumSquaredDiffs := 0.0
    
    for _, value := range s.Data {
        diff := value - mean
        sumSquaredDiffs += diff * diff
    }
    
    return sumSquaredDiffs / float64(s.N-1)
}

func (s *Statistics) StandardDeviation() float64 {
    return math.Sqrt(s.Variance())
}

func (s *Statistics) Range() float64 {
    if s.N == 0 {
        return 0
    }
    
    min := s.Data[0]
    max := s.Data[0]
    
    for _, value := range s.Data {
        if value < min {
            min = value
        }
        if value > max {
            max = value
        }
    }
    
    return max - min
}

func (s *Statistics) Quartiles() (float64, float64, float64) {
    if s.N == 0 {
        return 0, 0, 0
    }
    
    sorted := make([]float64, s.N)
    copy(sorted, s.Data)
    sort.Float64s(sorted)
    
    q1 := s.percentile(sorted, 25)
    q2 := s.percentile(sorted, 50)
    q3 := s.percentile(sorted, 75)
    
    return q1, q2, q3
}

func (s *Statistics) percentile(sorted []float64, p float64) float64 {
    if len(sorted) == 0 {
        return 0
    }
    
    index := p / 100.0 * float64(len(sorted)-1)
    
    if index == float64(int(index)) {
        return sorted[int(index)]
    }
    
    lower := int(index)
    upper := lower + 1
    
    if upper >= len(sorted) {
        return sorted[len(sorted)-1]
    }
    
    weight := index - float64(lower)
    return sorted[lower]*(1-weight) + sorted[upper]*weight
}

func (s *Statistics) InterquartileRange() float64 {
    q1, _, q3 := s.Quartiles()
    return q3 - q1
}

func (s *Statistics) Skewness() float64 {
    if s.N <= 1 {
        return 0
    }
    
    mean := s.Mean()
    std := s.StandardDeviation()
    
    if std == 0 {
        return 0
    }
    
    sum := 0.0
    for _, value := range s.Data {
        normalized := (value - mean) / std
        sum += normalized * normalized * normalized
    }
    
    return sum / float64(s.N)
}

func (s *Statistics) Kurtosis() float64 {
    if s.N <= 1 {
        return 0
    }
    
    mean := s.Mean()
    std := s.StandardDeviation()
    
    if std == 0 {
        return 0
    }
    
    sum := 0.0
    for _, value := range s.Data {
        normalized := (value - mean) / std
        sum += normalized * normalized * normalized * normalized
    }
    
    return sum/float64(s.N) - 3.0
}

func (s *Statistics) Summary() map[string]float64 {
    return map[string]float64{
        "Count":           float64(s.N),
        "Mean":            s.Mean(),
        "Median":          s.Median(),
        "Mode":            s.Mode()[0], // First mode if multiple
        "StandardDev":     s.StandardDeviation(),
        "Variance":        s.Variance(),
        "Min":             s.Data[0], // Assuming sorted
        "Max":             s.Data[s.N-1],
        "Range":           s.Range(),
        "Skewness":        s.Skewness(),
        "Kurtosis":        s.Kurtosis(),
    }
}

func main() {
    data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    stats := NewStatistics(data)
    
    fmt.Println("Data:", data)
    fmt.Println("Mean:", stats.Mean())
    fmt.Println("Median:", stats.Median())
    fmt.Println("Mode:", stats.Mode())
    fmt.Println("Standard Deviation:", stats.StandardDeviation())
    fmt.Println("Variance:", stats.Variance())
    fmt.Println("Range:", stats.Range())
    
    q1, q2, q3 := stats.Quartiles()
    fmt.Printf("Quartiles: Q1=%.2f, Q2=%.2f, Q3=%.2f\n", q1, q2, q3)
    fmt.Println("IQR:", stats.InterquartileRange())
    fmt.Println("Skewness:", stats.Skewness())
    fmt.Println("Kurtosis:", stats.Kurtosis())
}
```

#### Node.js Implementation

```javascript
class Statistics {
    constructor(data) {
        this.data = data;
        this.n = data.length;
    }
    
    mean() {
        if (this.n === 0) return 0;
        return this.data.reduce((sum, value) => sum + value, 0) / this.n;
    }
    
    median() {
        if (this.n === 0) return 0;
        
        const sorted = [...this.data].sort((a, b) => a - b);
        
        if (this.n % 2 === 0) {
            return (sorted[this.n / 2 - 1] + sorted[this.n / 2]) / 2;
        }
        return sorted[Math.floor(this.n / 2)];
    }
    
    mode() {
        if (this.n === 0) return [];
        
        const frequency = {};
        for (const value of this.data) {
            frequency[value] = (frequency[value] || 0) + 1;
        }
        
        const maxFreq = Math.max(...Object.values(frequency));
        return Object.keys(frequency)
            .filter(key => frequency[key] === maxFreq)
            .map(Number);
    }
    
    variance() {
        if (this.n <= 1) return 0;
        
        const mean = this.mean();
        const sumSquaredDiffs = this.data.reduce((sum, value) => {
            const diff = value - mean;
            return sum + diff * diff;
        }, 0);
        
        return sumSquaredDiffs / (this.n - 1);
    }
    
    standardDeviation() {
        return Math.sqrt(this.variance());
    }
    
    range() {
        if (this.n === 0) return 0;
        return Math.max(...this.data) - Math.min(...this.data);
    }
    
    quartiles() {
        if (this.n === 0) return { q1: 0, q2: 0, q3: 0 };
        
        const sorted = [...this.data].sort((a, b) => a - b);
        
        return {
            q1: this.percentile(sorted, 25),
            q2: this.percentile(sorted, 50),
            q3: this.percentile(sorted, 75)
        };
    }
    
    percentile(sorted, p) {
        if (sorted.length === 0) return 0;
        
        const index = p / 100 * (sorted.length - 1);
        
        if (index === Math.floor(index)) {
            return sorted[index];
        }
        
        const lower = Math.floor(index);
        const upper = lower + 1;
        
        if (upper >= sorted.length) {
            return sorted[sorted.length - 1];
        }
        
        const weight = index - lower;
        return sorted[lower] * (1 - weight) + sorted[upper] * weight;
    }
    
    interquartileRange() {
        const { q1, q3 } = this.quartiles();
        return q3 - q1;
    }
    
    skewness() {
        if (this.n <= 1) return 0;
        
        const mean = this.mean();
        const std = this.standardDeviation();
        
        if (std === 0) return 0;
        
        const sum = this.data.reduce((sum, value) => {
            const normalized = (value - mean) / std;
            return sum + normalized * normalized * normalized;
        }, 0);
        
        return sum / this.n;
    }
    
    kurtosis() {
        if (this.n <= 1) return 0;
        
        const mean = this.mean();
        const std = this.standardDeviation();
        
        if (std === 0) return 0;
        
        const sum = this.data.reduce((sum, value) => {
            const normalized = (value - mean) / std;
            return sum + normalized * normalized * normalized * normalized;
        }, 0);
        
        return sum / this.n - 3;
    }
    
    summary() {
        return {
            count: this.n,
            mean: this.mean(),
            median: this.median(),
            mode: this.mode()[0] || 0,
            standardDev: this.standardDeviation(),
            variance: this.variance(),
            min: Math.min(...this.data),
            max: Math.max(...this.data),
            range: this.range(),
            skewness: this.skewness(),
            kurtosis: this.kurtosis()
        };
    }
}

// Example usage
const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const stats = new Statistics(data);

console.log('Data:', data);
console.log('Mean:', stats.mean());
console.log('Median:', stats.median());
console.log('Mode:', stats.mode());
console.log('Standard Deviation:', stats.standardDeviation());
console.log('Variance:', stats.variance());
console.log('Range:', stats.range());

const { q1, q2, q3 } = stats.quartiles();
console.log(`Quartiles: Q1=${q1.toFixed(2)}, Q2=${q2.toFixed(2)}, Q3=${q3.toFixed(2)}`);
console.log('IQR:', stats.interquartileRange());
console.log('Skewness:', stats.skewness());
console.log('Kurtosis:', stats.kurtosis());
```

## Probability Theory

### 1. Basic Probability Concepts

#### Probability Calculations

```go
package main

import (
    "fmt"
    "math"
)

type Probability struct{}

func NewProbability() *Probability {
    return &Probability{}
}

func (p *Probability) Factorial(n int) int64 {
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

func (p *Probability) Permutation(n, r int) int64 {
    if n < 0 || r < 0 || r > n {
        return 0
    }
    return p.Factorial(n) / p.Factorial(n-r)
}

func (p *Probability) Combination(n, r int) int64 {
    if n < 0 || r < 0 || r > n {
        return 0
    }
    return p.Factorial(n) / (p.Factorial(r) * p.Factorial(n-r))
}

func (p *Probability) BinomialProbability(n int, k int, p_success float64) float64 {
    if n < 0 || k < 0 || k > n {
        return 0
    }
    
    combination := float64(p.Combination(n, k))
    success := math.Pow(p_success, float64(k))
    failure := math.Pow(1-p_success, float64(n-k))
    
    return combination * success * failure
}

func (p *Probability) ConditionalProbability(pA, pB, pAB float64) float64 {
    if pB == 0 {
        return 0
    }
    return pAB / pB
}

func (p *Probability) BayesTheorem(pA, pB, pBA float64) float64 {
    if pB == 0 {
        return 0
    }
    return (pBA * pA) / pB
}

func (p *Probability) IndependenceTest(pA, pB, pAB float64) bool {
    expected := pA * pB
    return math.Abs(pAB - expected) < 1e-10
}

// Example usage
func main() {
    prob := NewProbability()
    
    fmt.Println("Factorial of 5:", prob.Factorial(5))
    fmt.Println("Permutation P(10,3):", prob.Permutation(10, 3))
    fmt.Println("Combination C(10,3):", prob.Combination(10, 3))
    
    fmt.Println("Binomial probability (n=10, k=3, p=0.5):", 
                prob.BinomialProbability(10, 3, 0.5))
    
    fmt.Println("Conditional probability P(A|B):", 
                prob.ConditionalProbability(0.3, 0.4, 0.2))
    
    fmt.Println("Bayes theorem P(A|B):", 
                prob.BayesTheorem(0.3, 0.4, 0.2))
    
    fmt.Println("Independence test:", 
                prob.IndependenceTest(0.3, 0.4, 0.12))
}
```

## Probability Distributions

### 1. Common Distributions

#### Discrete and Continuous Distributions

```go
package main

import (
    "fmt"
    "math"
)

type Distribution struct{}

func NewDistribution() *Distribution {
    return &Distribution{}
}

// Normal Distribution
func (d *Distribution) NormalPDF(x, mean, std float64) float64 {
    variance := std * std
    coefficient := 1.0 / math.Sqrt(2*math.Pi*variance)
    exponent := -((x-mean)*(x-mean)) / (2*variance)
    return coefficient * math.Exp(exponent)
}

func (d *Distribution) NormalCDF(x, mean, std float64) float64 {
    // Approximation using error function
    z := (x - mean) / std
    return 0.5 * (1 + d.erf(z/math.Sqrt2))
}

func (d *Distribution) erf(x float64) float64 {
    // Approximation of error function
    a1 := 0.254829592
    a2 := -0.284496736
    a3 := 1.421413741
    a4 := -1.453152027
    a5 := 1.061405429
    p := 0.3275911
    
    sign := 1.0
    if x < 0 {
        sign = -1.0
    }
    x = math.Abs(x)
    
    t := 1.0 / (1.0 + p*x)
    y := 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.Exp(-x*x)
    
    return sign * y
}

// Binomial Distribution
func (d *Distribution) BinomialPMF(n int, k int, p float64) float64 {
    if n < 0 || k < 0 || k > n {
        return 0
    }
    
    combination := d.combination(n, k)
    return combination * math.Pow(p, float64(k)) * math.Pow(1-p, float64(n-k))
}

func (d *Distribution) BinomialCDF(n int, k int, p float64) float64 {
    sum := 0.0
    for i := 0; i <= k; i++ {
        sum += d.BinomialPMF(n, i, p)
    }
    return sum
}

func (d *Distribution) combination(n, k int) float64 {
    if k > n-k {
        k = n - k
    }
    
    result := 1.0
    for i := 0; i < k; i++ {
        result = result * float64(n-i) / float64(i+1)
    }
    return result
}

// Poisson Distribution
func (d *Distribution) PoissonPMF(k int, lambda float64) float64 {
    if k < 0 || lambda <= 0 {
        return 0
    }
    
    return math.Exp(-lambda) * math.Pow(lambda, float64(k)) / d.factorial(k)
}

func (d *Distribution) PoissonCDF(k int, lambda float64) float64 {
    sum := 0.0
    for i := 0; i <= k; i++ {
        sum += d.PoissonPMF(i, lambda)
    }
    return sum
}

func (d *Distribution) factorial(n int) float64 {
    if n <= 1 {
        return 1
    }
    result := 1.0
    for i := 2; i <= n; i++ {
        result *= float64(i)
    }
    return result
}

// Exponential Distribution
func (d *Distribution) ExponentialPDF(x, lambda float64) float64 {
    if x < 0 || lambda <= 0 {
        return 0
    }
    return lambda * math.Exp(-lambda*x)
}

func (d *Distribution) ExponentialCDF(x, lambda float64) float64 {
    if x < 0 || lambda <= 0 {
        return 0
    }
    return 1 - math.Exp(-lambda*x)
}

// Uniform Distribution
func (d *Distribution) UniformPDF(x, a, b float64) float64 {
    if x < a || x > b {
        return 0
    }
    return 1.0 / (b - a)
}

func (d *Distribution) UniformCDF(x, a, b float64) float64 {
    if x < a {
        return 0
    }
    if x > b {
        return 1
    }
    return (x - a) / (b - a)
}

func main() {
    dist := NewDistribution()
    
    fmt.Println("Normal PDF at x=0, mean=0, std=1:", dist.NormalPDF(0, 0, 1))
    fmt.Println("Normal CDF at x=0, mean=0, std=1:", dist.NormalCDF(0, 0, 1))
    
    fmt.Println("Binomial PMF (n=10, k=3, p=0.5):", dist.BinomialPMF(10, 3, 0.5))
    fmt.Println("Binomial CDF (n=10, k=3, p=0.5):", dist.BinomialCDF(10, 3, 0.5))
    
    fmt.Println("Poisson PMF (k=3, lambda=2):", dist.PoissonPMF(3, 2))
    fmt.Println("Poisson CDF (k=3, lambda=2):", dist.PoissonCDF(3, 2))
    
    fmt.Println("Exponential PDF (x=1, lambda=2):", dist.ExponentialPDF(1, 2))
    fmt.Println("Exponential CDF (x=1, lambda=2):", dist.ExponentialCDF(1, 2))
    
    fmt.Println("Uniform PDF (x=0.5, a=0, b=1):", dist.UniformPDF(0.5, 0, 1))
    fmt.Println("Uniform CDF (x=0.5, a=0, b=1):", dist.UniformCDF(0.5, 0, 1))
}
```

## Hypothesis Testing

### 1. Statistical Tests

#### T-test and Chi-square Test

```go
package main

import (
    "fmt"
    "math"
)

type HypothesisTest struct{}

func NewHypothesisTest() *HypothesisTest {
    return &HypothesisTest{}
}

// One-sample t-test
func (ht *HypothesisTest) OneSampleTTest(data []float64, populationMean float64) (float64, float64, bool) {
    n := len(data)
    if n <= 1 {
        return 0, 0, false
    }
    
    // Calculate sample statistics
    sampleMean := ht.mean(data)
    sampleStd := ht.standardDeviation(data)
    
    // Calculate t-statistic
    t := (sampleMean - populationMean) / (sampleStd / math.Sqrt(float64(n)))
    
    // Calculate degrees of freedom
    df := n - 1
    
    // Calculate p-value (approximation)
    pValue := ht.tTestPValue(t, df)
    
    // Reject null hypothesis if p < 0.05
    reject := pValue < 0.05
    
    return t, pValue, reject
}

// Two-sample t-test
func (ht *HypothesisTest) TwoSampleTTest(data1, data2 []float64) (float64, float64, bool) {
    n1, n2 := len(data1), len(data2)
    if n1 <= 1 || n2 <= 1 {
        return 0, 0, false
    }
    
    mean1 := ht.mean(data1)
    mean2 := ht.mean(data2)
    std1 := ht.standardDeviation(data1)
    std2 := ht.standardDeviation(data2)
    
    // Pooled standard error
    pooledStd := math.Sqrt(((float64(n1)-1)*std1*std1 + (float64(n2)-1)*std2*std2) / 
                          (float64(n1+n2-2)))
    
    // Standard error of difference
    seDiff := pooledStd * math.Sqrt(1.0/float64(n1) + 1.0/float64(n2))
    
    // t-statistic
    t := (mean1 - mean2) / seDiff
    
    // Degrees of freedom
    df := n1 + n2 - 2
    
    // p-value
    pValue := ht.tTestPValue(t, df)
    
    reject := pValue < 0.05
    
    return t, pValue, reject
}

// Chi-square test for independence
func (ht *HypothesisTest) ChiSquareTest(observed [][]float64) (float64, float64, bool) {
    rows := len(observed)
    cols := len(observed[0])
    
    // Calculate row and column totals
    rowTotals := make([]float64, rows)
    colTotals := make([]float64, cols)
    grandTotal := 0.0
    
    for i := 0; i < rows; i++ {
        for j := 0; j < cols; j++ {
            rowTotals[i] += observed[i][j]
            colTotals[j] += observed[i][j]
            grandTotal += observed[i][j]
        }
    }
    
    // Calculate expected frequencies
    expected := make([][]float64, rows)
    for i := 0; i < rows; i++ {
        expected[i] = make([]float64, cols)
        for j := 0; j < cols; j++ {
            expected[i][j] = rowTotals[i] * colTotals[j] / grandTotal
        }
    }
    
    // Calculate chi-square statistic
    chiSquare := 0.0
    for i := 0; i < rows; i++ {
        for j := 0; j < cols; j++ {
            if expected[i][j] > 0 {
                chiSquare += (observed[i][j] - expected[i][j]) * 
                           (observed[i][j] - expected[i][j]) / expected[i][j]
            }
        }
    }
    
    // Degrees of freedom
    df := (rows - 1) * (cols - 1)
    
    // p-value (approximation)
    pValue := ht.chiSquarePValue(chiSquare, df)
    
    reject := pValue < 0.05
    
    return chiSquare, pValue, reject
}

// Helper functions
func (ht *HypothesisTest) mean(data []float64) float64 {
    sum := 0.0
    for _, value := range data {
        sum += value
    }
    return sum / float64(len(data))
}

func (ht *HypothesisTest) standardDeviation(data []float64) float64 {
    n := len(data)
    if n <= 1 {
        return 0
    }
    
    mean := ht.mean(data)
    sumSquaredDiffs := 0.0
    
    for _, value := range data {
        diff := value - mean
        sumSquaredDiffs += diff * diff
    }
    
    return math.Sqrt(sumSquaredDiffs / float64(n-1))
}

func (ht *HypothesisTest) tTestPValue(t float64, df int) float64 {
    // Simplified p-value calculation
    // In practice, you'd use more sophisticated methods
    if df <= 0 {
        return 1.0
    }
    
    // Approximation for p-value
    absT := math.Abs(t)
    if absT > 3.0 {
        return 0.001
    } else if absT > 2.0 {
        return 0.05
    } else if absT > 1.0 {
        return 0.3
    }
    return 0.5
}

func (ht *HypothesisTest) chiSquarePValue(chiSquare float64, df int) float64 {
    // Simplified p-value calculation
    if df <= 0 {
        return 1.0
    }
    
    // Approximation for p-value
    if chiSquare > 20.0 {
        return 0.001
    } else if chiSquare > 10.0 {
        return 0.01
    } else if chiSquare > 5.0 {
        return 0.05
    }
    return 0.5
}

func main() {
    test := NewHypothesisTest()
    
    // One-sample t-test
    data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    t, p, reject := test.OneSampleTTest(data, 5.0)
    fmt.Printf("One-sample t-test: t=%.3f, p=%.3f, reject=%t\n", t, p, reject)
    
    // Two-sample t-test
    data1 := []float64{1, 2, 3, 4, 5}
    data2 := []float64{6, 7, 8, 9, 10}
    t2, p2, reject2 := test.TwoSampleTTest(data1, data2)
    fmt.Printf("Two-sample t-test: t=%.3f, p=%.3f, reject=%t\n", t2, p2, reject2)
    
    // Chi-square test
    observed := [][]float64{
        {10, 20, 30},
        {15, 25, 35},
    }
    chi, p3, reject3 := test.ChiSquareTest(observed)
    fmt.Printf("Chi-square test: χ²=%.3f, p=%.3f, reject=%t\n", chi, p3, reject3)
}
```

## Regression Analysis

### 1. Linear Regression

#### Simple and Multiple Linear Regression

```go
package main

import (
    "fmt"
    "math"
)

type LinearRegression struct {
    coefficients []float64
    intercept    float64
    rSquared     float64
}

func NewLinearRegression() *LinearRegression {
    return &LinearRegression{}
}

func (lr *LinearRegression) Fit(X [][]float64, y []float64) {
    n := len(X)
    if n == 0 {
        return
    }
    
    p := len(X[0]) // number of features
    
    // Add intercept term
    XWithIntercept := make([][]float64, n)
    for i := 0; i < n; i++ {
        XWithIntercept[i] = make([]float64, p+1)
        XWithIntercept[i][0] = 1.0 // intercept
        copy(XWithIntercept[i][1:], X[i])
    }
    
    // Solve normal equations: (X'X)^-1 X'y
    XTX := lr.matrixMultiply(lr.transpose(XWithIntercept), XWithIntercept)
    XTy := lr.matrixVectorMultiply(lr.transpose(XWithIntercept), y)
    
    // Solve linear system
    coefficients := lr.solveLinearSystem(XTX, XTy)
    
    lr.intercept = coefficients[0]
    lr.coefficients = coefficients[1:]
    
    // Calculate R-squared
    lr.rSquared = lr.calculateRSquared(X, y)
}

func (lr *LinearRegression) Predict(X [][]float64) []float64 {
    predictions := make([]float64, len(X))
    
    for i, x := range X {
        prediction := lr.intercept
        for j, coef := range lr.coefficients {
            if j < len(x) {
                prediction += coef * x[j]
            }
        }
        predictions[i] = prediction
    }
    
    return predictions
}

func (lr *LinearRegression) calculateRSquared(X [][]float64, y []float64) float64 {
    predictions := lr.Predict(X)
    
    // Calculate mean of y
    yMean := 0.0
    for _, yi := range y {
        yMean += yi
    }
    yMean /= float64(len(y))
    
    // Calculate R-squared
    ssRes := 0.0 // residual sum of squares
    ssTot := 0.0 // total sum of squares
    
    for i := 0; i < len(y); i++ {
        ssRes += (y[i] - predictions[i]) * (y[i] - predictions[i])
        ssTot += (y[i] - yMean) * (y[i] - yMean)
    }
    
    if ssTot == 0 {
        return 0
    }
    
    return 1 - ssRes/ssTot
}

// Helper functions for matrix operations
func (lr *LinearRegression) transpose(matrix [][]float64) [][]float64 {
    rows := len(matrix)
    cols := len(matrix[0])
    
    result := make([][]float64, cols)
    for i := 0; i < cols; i++ {
        result[i] = make([]float64, rows)
        for j := 0; j < rows; j++ {
            result[i][j] = matrix[j][i]
        }
    }
    
    return result
}

func (lr *LinearRegression) matrixMultiply(A, B [][]float64) [][]float64 {
    rowsA := len(A)
    colsA := len(A[0])
    colsB := len(B[0])
    
    result := make([][]float64, rowsA)
    for i := 0; i < rowsA; i++ {
        result[i] = make([]float64, colsB)
        for j := 0; j < colsB; j++ {
            for k := 0; k < colsA; k++ {
                result[i][j] += A[i][k] * B[k][j]
            }
        }
    }
    
    return result
}

func (lr *LinearRegression) matrixVectorMultiply(A [][]float64, b []float64) []float64 {
    result := make([]float64, len(A))
    
    for i := 0; i < len(A); i++ {
        for j := 0; j < len(b); j++ {
            result[i] += A[i][j] * b[j]
        }
    }
    
    return result
}

func (lr *LinearRegression) solveLinearSystem(A [][]float64, b []float64) []float64 {
    n := len(b)
    
    // Gaussian elimination with partial pivoting
    for i := 0; i < n; i++ {
        // Find pivot
        maxRow := i
        for k := i + 1; k < n; k++ {
            if math.Abs(A[k][i]) > math.Abs(A[maxRow][i]) {
                maxRow = k
            }
        }
        
        // Swap rows
        A[i], A[maxRow] = A[maxRow], A[i]
        b[i], b[maxRow] = b[maxRow], b[i]
        
        // Make all rows below this one 0 in current column
        for k := i + 1; k < n; k++ {
            if A[i][i] != 0 {
                factor := A[k][i] / A[i][i]
                for j := i; j < n; j++ {
                    A[k][j] -= factor * A[i][j]
                }
                b[k] -= factor * b[i]
            }
        }
    }
    
    // Back substitution
    x := make([]float64, n)
    for i := n - 1; i >= 0; i-- {
        x[i] = b[i]
        for j := i + 1; j < n; j++ {
            x[i] -= A[i][j] * x[j]
        }
        if A[i][i] != 0 {
            x[i] /= A[i][i]
        }
    }
    
    return x
}

func main() {
    // Example data
    X := [][]float64{
        {1, 2},
        {2, 3},
        {3, 4},
        {4, 5},
        {5, 6},
    }
    y := []float64{3, 5, 7, 9, 11}
    
    // Create and fit model
    model := NewLinearRegression()
    model.Fit(X, y)
    
    fmt.Printf("Intercept: %.3f\n", model.intercept)
    fmt.Printf("Coefficients: %v\n", model.coefficients)
    fmt.Printf("R-squared: %.3f\n", model.rSquared)
    
    // Make predictions
    testX := [][]float64{{6, 7}, {7, 8}}
    predictions := model.Predict(testX)
    fmt.Printf("Predictions: %v\n", predictions)
}
```

## Follow-up Questions

### 1. Descriptive Statistics
**Q: When would you use median instead of mean?**
A: Use median when dealing with skewed data or outliers, as it's more robust to extreme values. The median represents the middle value and isn't affected by outliers.

### 2. Probability Distributions
**Q: What's the difference between PMF and PDF?**
A: PMF (Probability Mass Function) is for discrete random variables and gives the probability of exact values. PDF (Probability Density Function) is for continuous random variables and gives the density at a point.

### 3. Hypothesis Testing
**Q: What does a p-value tell us?**
A: A p-value tells us the probability of observing the data (or more extreme) if the null hypothesis is true. A small p-value (typically < 0.05) suggests we should reject the null hypothesis.

## Sources

### Books
- **Introduction to Statistical Learning** by James, Witten, Hastie, Tibshirani
- **The Elements of Statistical Learning** by Hastie, Tibshirani, Friedman
- **Probability and Statistics** by Morris DeGroot

### Online Resources
- **Khan Academy** - Statistics and probability
- **Coursera** - Statistics courses
- **MIT OpenCourseWare** - Introduction to probability

## Projects

### 1. Statistical Analysis Tool
**Objective**: Build a comprehensive statistical analysis tool
**Requirements**: Descriptive statistics, hypothesis testing, regression
**Deliverables**: Complete statistical analysis application

### 2. A/B Testing Framework
**Objective**: Create a framework for A/B testing
**Requirements**: Statistical tests, confidence intervals, power analysis
**Deliverables**: A/B testing platform with statistical validation

### 3. Data Visualization Dashboard
**Objective**: Build a dashboard for statistical data visualization
**Requirements**: Charts, distributions, correlation analysis
**Deliverables**: Interactive statistical visualization tool

---

**Next**: [Discrete Mathematics](discrete-mathematics.md/) | **Previous**: [Calculus](calculus.md/) | **Up**: [Phase 0](README.md/)

