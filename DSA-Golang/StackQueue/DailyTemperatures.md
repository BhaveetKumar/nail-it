# Daily Temperatures

### Problem
Given an array of integers `temperatures` representing the daily temperatures, return an array `answer` such that `answer[i]` is the number of days you have to wait after the `ith` day to get a warmer temperature. If there is no future day for which this is possible, keep `answer[i] == 0` instead.

**Example:**
```
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]

Input: temperatures = [30,40,50,60]
Output: [1,1,1,0]
```

### Golang Solution

```go
func dailyTemperatures(temperatures []int) []int {
    n := len(temperatures)
    result := make([]int, n)
    stack := []int{} // Stack to store indices
    
    for i := 0; i < n; i++ {
        for len(stack) > 0 && temperatures[i] > temperatures[stack[len(stack)-1]] {
            index := stack[len(stack)-1]
            stack = stack[:len(stack)-1]
            result[index] = i - index
        }
        stack = append(stack, i)
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
