# Combination Sum

### Problem
Given an array of distinct integers `candidates` and a target integer `target`, return a list of all unique combinations of `candidates` where the chosen numbers sum to `target`. You may return the combinations in any order.

The same number may be chosen from `candidates` an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

**Example:**
```
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.

Input: candidates = [2,3,5], target = 8
Output: [[2,2,2,2],[2,3,3],[3,5]]
```

### Golang Solution

```go
func combinationSum(candidates []int, target int) [][]int {
    var result [][]int
    var current []int
    
    var backtrack func(int, int)
    backtrack = func(start int, remaining int) {
        if remaining == 0 {
            combination := make([]int, len(current))
            copy(combination, current)
            result = append(result, combination)
            return
        }
        
        if remaining < 0 {
            return
        }
        
        for i := start; i < len(candidates); i++ {
            current = append(current, candidates[i])
            backtrack(i, remaining-candidates[i])
            current = current[:len(current)-1]
        }
    }
    
    backtrack(0, target)
    return result
}
```

### Alternative Solutions

#### **Using Sort for Optimization**
```go
import "sort"

func combinationSumSorted(candidates []int, target int) [][]int {
    sort.Ints(candidates)
    var result [][]int
    var current []int
    
    var backtrack func(int, int)
    backtrack = func(start int, remaining int) {
        if remaining == 0 {
            combination := make([]int, len(current))
            copy(combination, current)
            result = append(result, combination)
            return
        }
        
        for i := start; i < len(candidates); i++ {
            if candidates[i] > remaining {
                break
            }
            
            current = append(current, candidates[i])
            backtrack(i, remaining-candidates[i])
            current = current[:len(current)-1]
        }
    }
    
    backtrack(0, target)
    return result
}
```

#### **Iterative Approach**
```go
func combinationSumIterative(candidates []int, target int) [][]int {
    var result [][]int
    stack := []State{{start: 0, path: []int{}, remaining: target}}
    
    for len(stack) > 0 {
        current := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        
        if current.remaining == 0 {
            result = append(result, current.path)
            continue
        }
        
        if current.remaining < 0 {
            continue
        }
        
        for i := current.start; i < len(candidates); i++ {
            newPath := make([]int, len(current.path))
            copy(newPath, current.path)
            newPath = append(newPath, candidates[i])
            
            stack = append(stack, State{
                start:     i,
                path:      newPath,
                remaining: current.remaining - candidates[i],
            })
        }
    }
    
    return result
}

type State struct {
    start     int
    path      []int
    remaining int
}
```

#### **Using Memoization**
```go
func combinationSumMemo(candidates []int, target int) [][]int {
    memo := make(map[int][][]int)
    return combinationSumHelper(candidates, target, 0, memo)
}

func combinationSumHelper(candidates []int, target, start int, memo map[int][][]int) [][]int {
    if target == 0 {
        return [][]int{{}}
    }
    
    if target < 0 {
        return [][]int{}
    }
    
    key := target*1000 + start
    if result, exists := memo[key]; exists {
        return result
    }
    
    var result [][]int
    
    for i := start; i < len(candidates); i++ {
        if candidates[i] <= target {
            subResults := combinationSumHelper(candidates, target-candidates[i], i, memo)
            
            for _, subResult := range subResults {
                newResult := make([]int, len(subResult)+1)
                newResult[0] = candidates[i]
                copy(newResult[1:], subResult)
                result = append(result, newResult)
            }
        }
    }
    
    memo[key] = result
    return result
}
```

#### **Return with Count**
```go
func combinationSumWithCount(candidates []int, target int) ([][]int, int) {
    var result [][]int
    var current []int
    
    var backtrack func(int, int)
    backtrack = func(start int, remaining int) {
        if remaining == 0 {
            combination := make([]int, len(current))
            copy(combination, current)
            result = append(result, combination)
            return
        }
        
        if remaining < 0 {
            return
        }
        
        for i := start; i < len(candidates); i++ {
            current = append(current, candidates[i])
            backtrack(i, remaining-candidates[i])
            current = current[:len(current)-1]
        }
    }
    
    backtrack(0, target)
    return result, len(result)
}
```

#### **Using Bit Manipulation**
```go
func combinationSumBitManipulation(candidates []int, target int) [][]int {
    var result [][]int
    n := len(candidates)
    
    // Generate all possible combinations using bit manipulation
    for mask := 0; mask < (1 << (n * 4)); mask++ { // Assuming max 4 repetitions
        var combination []int
        sum := 0
        
        for i := 0; i < n; i++ {
            count := (mask >> (i * 4)) & 15 // 4 bits per candidate
            for j := 0; j < count; j++ {
                combination = append(combination, candidates[i])
                sum += candidates[i]
            }
        }
        
        if sum == target {
            result = append(result, combination)
        }
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(2^target) in worst case
- **Space Complexity:** O(target) for recursion depth