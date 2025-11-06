---
# Auto-generated front matter
Title: Dailytemperatures
LastUpdated: 2025-11-06T20:45:58.703159
Tags: []
Status: draft
---

# Daily Temperatures

### Problem
Given an array of integers `temperatures` representing the daily temperatures, return an array `answer` such that `answer[i]` is the number of days you have to wait after the `ith` day to get a warmer temperature. If there is no future day for which this is possible, keep `answer[i] == 0` instead.

**Example:**
```
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]

Input: temperatures = [30,40,50,60]
Output: [1,1,1,0]

Input: temperatures = [30,60,90]
Output: [1,1,0]
```

### Golang Solution

```go
func dailyTemperatures(temperatures []int) []int {
    n := len(temperatures)
    result := make([]int, n)
    stack := []int{} // Store indices
    
    for i := 0; i < n; i++ {
        // While stack is not empty and current temperature is warmer
        for len(stack) > 0 && temperatures[stack[len(stack)-1]] < temperatures[i] {
            prevIndex := stack[len(stack)-1]
            stack = stack[:len(stack)-1]
            result[prevIndex] = i - prevIndex
        }
        
        stack = append(stack, i)
    }
    
    return result
}
```

### Alternative Solutions

#### **Using Two Pointers**
```go
func dailyTemperaturesTwoPointers(temperatures []int) []int {
    n := len(temperatures)
    result := make([]int, n)
    
    for i := 0; i < n; i++ {
        for j := i + 1; j < n; j++ {
            if temperatures[j] > temperatures[i] {
                result[i] = j - i
                break
            }
        }
    }
    
    return result
}
```

#### **Using Array as Stack**
```go
func dailyTemperaturesArray(temperatures []int) []int {
    n := len(temperatures)
    result := make([]int, n)
    stack := make([]int, n)
    top := -1
    
    for i := 0; i < n; i++ {
        for top >= 0 && temperatures[stack[top]] < temperatures[i] {
            prevIndex := stack[top]
            top--
            result[prevIndex] = i - prevIndex
        }
        
        top++
        stack[top] = i
    }
    
    return result
}
```

#### **Return with Temperature Info**
```go
type TemperatureResult struct {
    Days        int
    Temperature int
    Index       int
}

func dailyTemperaturesWithInfo(temperatures []int) []TemperatureResult {
    n := len(temperatures)
    result := make([]TemperatureResult, n)
    stack := []int{} // Store indices
    
    for i := 0; i < n; i++ {
        result[i] = TemperatureResult{
            Days:        0,
            Temperature: temperatures[i],
            Index:       i,
        }
        
        for len(stack) > 0 && temperatures[stack[len(stack)-1]] < temperatures[i] {
            prevIndex := stack[len(stack)-1]
            stack = stack[:len(stack)-1]
            result[prevIndex].Days = i - prevIndex
        }
        
        stack = append(stack, i)
    }
    
    return result
}
```

#### **Return All Warmer Days**
```go
func dailyTemperaturesAllWarmer(temperatures []int) [][]int {
    n := len(temperatures)
    result := make([][]int, n)
    
    for i := 0; i < n; i++ {
        var warmerDays []int
        
        for j := i + 1; j < n; j++ {
            if temperatures[j] > temperatures[i] {
                warmerDays = append(warmerDays, j)
            }
        }
        
        result[i] = warmerDays
    }
    
    return result
}
```

#### **Return Statistics**
```go
type TemperatureStats struct {
    MaxWaitDays    int
    MinWaitDays    int
    AvgWaitDays    float64
    NoWarmerDays   int
    TotalDays      int
}

func temperatureStatistics(temperatures []int) TemperatureStats {
    n := len(temperatures)
    waitDays := dailyTemperatures(temperatures)
    
    maxWait := 0
    minWait := math.MaxInt32
    sum := 0
    noWarmer := 0
    
    for _, days := range waitDays {
        if days > maxWait {
            maxWait = days
        }
        if days < minWait && days > 0 {
            minWait = days
        }
        sum += days
        if days == 0 {
            noWarmer++
        }
    }
    
    if minWait == math.MaxInt32 {
        minWait = 0
    }
    
    return TemperatureStats{
        MaxWaitDays:  maxWait,
        MinWaitDays:  minWait,
        AvgWaitDays:  float64(sum) / float64(n),
        NoWarmerDays: noWarmer,
        TotalDays:    n,
    }
}
```

#### **Return Temperature Trends**
```go
type TemperatureTrend struct {
    Increasing bool
    Decreasing bool
    Stable     bool
    Trend      string
}

func temperatureTrends(temperatures []int) []TemperatureTrend {
    n := len(temperatures)
    trends := make([]TemperatureTrend, n)
    
    for i := 0; i < n; i++ {
        trend := TemperatureTrend{}
        
        if i > 0 {
            if temperatures[i] > temperatures[i-1] {
                trend.Increasing = true
                trend.Trend = "increasing"
            } else if temperatures[i] < temperatures[i-1] {
                trend.Decreasing = true
                trend.Trend = "decreasing"
            } else {
                trend.Stable = true
                trend.Trend = "stable"
            }
        } else {
            trend.Trend = "first day"
        }
        
        trends[i] = trend
    }
    
    return trends
}
```

### Complexity
- **Time Complexity:** O(n) for stack approach, O(n²) for two pointers
- **Space Complexity:** O(n) for stack, O(1) for two pointers