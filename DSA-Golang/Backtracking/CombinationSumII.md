# Combination Sum II

### Problem
Given a collection of candidate numbers (`candidates`) and a target number (`target`), find all unique combinations in `candidates` where the candidate numbers sum to `target`.

Each number in `candidates` may only be used once in the combination.

Note: The solution set must not contain duplicate combinations.

**Example:**
```
Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: [[1,1,6],[1,2,5],[1,7],[2,6]]

Input: candidates = [2,5,2,1,2], target = 5
Output: [[1,2,2],[5]]
```

### Golang Solution

```go
import "sort"

func combinationSum2(candidates []int, target int) [][]int {
    sort.Ints(candidates)
    var result [][]int
    var backtrack func(int, []int, int)
    
    backtrack = func(start int, current []int, sum int) {
        if sum == target {
            combination := make([]int, len(current))
            copy(combination, current)
            result = append(result, combination)
            return
        }
        
        if sum > target {
            return
        }
        
        for i := start; i < len(candidates); i++ {
            // Skip duplicates
            if i > start && candidates[i] == candidates[i-1] {
                continue
            }
            
            current = append(current, candidates[i])
            backtrack(i+1, current, sum+candidates[i])
            current = current[:len(current)-1]
        }
    }
    
    backtrack(0, []int{}, 0)
    return result
}
```

### Alternative Solutions

#### **Using Set to Avoid Duplicates**
```go
func combinationSum2Set(candidates []int, target int) [][]int {
    sort.Ints(candidates)
    var result [][]int
    seen := make(map[string]bool)
    
    var backtrack func(int, []int, int)
    backtrack = func(start int, current []int, sum int) {
        if sum == target {
            key := fmt.Sprintf("%v", current)
            if !seen[key] {
                combination := make([]int, len(current))
                copy(combination, current)
                result = append(result, combination)
                seen[key] = true
            }
            return
        }
        
        if sum > target {
            return
        }
        
        for i := start; i < len(candidates); i++ {
            current = append(current, candidates[i])
            backtrack(i+1, current, sum+candidates[i])
            current = current[:len(current)-1]
        }
    }
    
    backtrack(0, []int{}, 0)
    return result
}
```

#### **Iterative Approach**
```go
func combinationSum2Iterative(candidates []int, target int) [][]int {
    sort.Ints(candidates)
    var result [][]int
    
    type State struct {
        index int
        path  []int
        sum   int
    }
    
    stack := []State{{index: 0, path: []int{}, sum: 0}}
    
    for len(stack) > 0 {
        current := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        
        if current.sum == target {
            combination := make([]int, len(current.path))
            copy(combination, current.path)
            result = append(result, combination)
            continue
        }
        
        if current.sum > target {
            continue
        }
        
        for i := current.index; i < len(candidates); i++ {
            if i > current.index && candidates[i] == candidates[i-1] {
                continue
            }
            
            newPath := make([]int, len(current.path))
            copy(newPath, current.path)
            newPath = append(newPath, candidates[i])
            
            stack = append(stack, State{
                index: i + 1,
                path:  newPath,
                sum:   current.sum + candidates[i],
            })
        }
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(2^n)
- **Space Complexity:** O(target)
