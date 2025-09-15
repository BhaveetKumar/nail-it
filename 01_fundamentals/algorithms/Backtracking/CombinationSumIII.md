# Combination Sum III

### Problem
Find all valid combinations of `k` numbers that sum up to `n` such that the following conditions are true:

- Only numbers 1 through 9 are used.
- Each number is used at most once.

Return a list of all possible valid combinations. The list must not contain the same combination twice, and the combinations may be returned in any order.

**Example:**
```
Input: k = 3, n = 7
Output: [[1,2,4]]
Explanation:
1 + 2 + 4 = 7
There are no other valid combinations.

Input: k = 3, n = 9
Output: [[1,2,6],[1,3,5],[2,3,4]]
Explanation:
1 + 2 + 6 = 9
1 + 3 + 5 = 9
2 + 3 + 4 = 9
There are no other valid combinations.
```

### Golang Solution

```go
func combinationSum3(k int, n int) [][]int {
    var result [][]int
    var backtrack func(int, []int, int)
    
    backtrack = func(start int, current []int, sum int) {
        if len(current) == k && sum == n {
            combination := make([]int, len(current))
            copy(combination, current)
            result = append(result, combination)
            return
        }
        
        if len(current) >= k || sum > n {
            return
        }
        
        for i := start; i <= 9; i++ {
            if sum+i > n {
                break
            }
            
            current = append(current, i)
            backtrack(i+1, current, sum+i)
            current = current[:len(current)-1]
        }
    }
    
    backtrack(1, []int{}, 0)
    return result
}
```

### Alternative Solutions

#### **Using Bit Manipulation**
```go
func combinationSum3BitManipulation(k int, n int) [][]int {
    var result [][]int
    
    // Try all possible combinations using bit manipulation
    for mask := 0; mask < (1 << 9); mask++ {
        if countBits(mask) != k {
            continue
        }
        
        combination := []int{}
        sum := 0
        
        for i := 0; i < 9; i++ {
            if mask&(1<<i) != 0 {
                num := i + 1
                combination = append(combination, num)
                sum += num
            }
        }
        
        if sum == n {
            result = append(result, combination)
        }
    }
    
    return result
}

func countBits(mask int) int {
    count := 0
    for mask > 0 {
        count += mask & 1
        mask >>= 1
    }
    return count
}
```

#### **Iterative Approach**
```go
func combinationSum3Iterative(k int, n int) [][]int {
    var result [][]int
    stack := []State{{start: 1, path: []int{}, sum: 0}}
    
    for len(stack) > 0 {
        current := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        
        if len(current.path) == k && current.sum == n {
            combination := make([]int, len(current.path))
            copy(combination, current.path)
            result = append(result, combination)
            continue
        }
        
        if len(current.path) >= k || current.sum > n {
            continue
        }
        
        for i := current.start; i <= 9; i++ {
            if current.sum+i > n {
                break
            }
            
            newPath := make([]int, len(current.path))
            copy(newPath, current.path)
            newPath = append(newPath, i)
            
            stack = append(stack, State{
                start: i + 1,
                path:  newPath,
                sum:   current.sum + i,
            })
        }
    }
    
    return result
}

type State struct {
    start int
    path  []int
    sum   int
}
```

#### **Mathematical Optimization**
```go
func combinationSum3Optimized(k int, n int) [][]int {
    var result [][]int
    
    // Early termination if impossible
    minSum := k * (k + 1) / 2
    maxSum := k * (19 - k) / 2
    
    if n < minSum || n > maxSum {
        return result
    }
    
    var backtrack func(int, []int, int)
    backtrack = func(start int, current []int, sum int) {
        if len(current) == k && sum == n {
            combination := make([]int, len(current))
            copy(combination, current)
            result = append(result, combination)
            return
        }
        
        if len(current) >= k || sum > n {
            return
        }
        
        // Calculate remaining numbers needed
        remaining := k - len(current)
        maxPossible := 9
        
        for i := start; i <= maxPossible; i++ {
            // Check if it's possible to reach target with remaining numbers
            if sum+i > n {
                break
            }
            
            // Check if we can reach target with remaining numbers
            minRemainingSum := (remaining - 1) * (i + 1 + i + remaining - 1) / 2
            maxRemainingSum := (remaining - 1) * (9 + 9 - remaining + 2) / 2
            
            if sum+i+minRemainingSum > n || sum+i+maxRemainingSum < n {
                continue
            }
            
            current = append(current, i)
            backtrack(i+1, current, sum+i)
            current = current[:len(current)-1]
        }
    }
    
    backtrack(1, []int{}, 0)
    return result
}
```

### Complexity
- **Time Complexity:** O(C(9,k)) where C(9,k) is combinations of k from 9
- **Space Complexity:** O(k) for recursion depth
