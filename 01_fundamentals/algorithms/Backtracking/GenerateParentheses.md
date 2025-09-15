# Generate Parentheses

### Problem
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

**Example:**
```
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]

Input: n = 1
Output: ["()"]
```

### Golang Solution

```go
func generateParenthesis(n int) []string {
    var result []string
    var backtrack func(string, int, int)
    
    backtrack = func(current string, open, close int) {
        if len(current) == 2*n {
            result = append(result, current)
            return
        }
        
        if open < n {
            backtrack(current+"(", open+1, close)
        }
        
        if close < open {
            backtrack(current+")", open, close+1)
        }
    }
    
    backtrack("", 0, 0)
    return result
}
```

### Complexity
- **Time Complexity:** O(4^n / √n)
- **Space Complexity:** O(4^n / √n)
