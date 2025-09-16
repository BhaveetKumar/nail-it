# Generate Parentheses

## Problem Statement

Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

**Example 1:**
```
Input: n = 3
Output: ["((()))", "(()())", "(())()", "()(())", "()()()"]
```

**Example 2:**
```
Input: n = 1
Output: ["()"]
```

## Approach

### Brute Force Approach
1. Generate all possible combinations of parentheses
2. Check which ones are valid
3. Return valid combinations

**Time Complexity:** O(2^(2n) × n) - Generate all combinations and validate each
**Space Complexity:** O(2^(2n) × n) - Store all combinations

### Optimized Backtracking Approach
1. Use backtracking to generate only valid combinations
2. Keep track of open and close parentheses count
3. Add '(' if we haven't used all open parentheses
4. Add ')' if we have more open than close parentheses

**Time Complexity:** O(4^n / √n) - Catalan number
**Space Complexity:** O(n) - Recursion depth

## Solution

```javascript
/**
 * Generate all valid combinations of n pairs of parentheses
 * @param {number} n - Number of pairs
 * @return {string[]} - Array of valid parentheses combinations
 */
function generateParenthesis(n) {
    const result = [];
    
    function backtrack(current, open, close) {
        // Base case: we've used all parentheses
        if (current.length === 2 * n) {
            result.push(current);
            return;
        }
        
        // Add opening parenthesis if we haven't used all
        if (open < n) {
            backtrack(current + '(', open + 1, close);
        }
        
        // Add closing parenthesis if we have more open than close
        if (close < open) {
            backtrack(current + ')', open, close + 1);
        }
    }
    
    backtrack('', 0, 0);
    return result;
}

// Alternative implementation with array manipulation
function generateParenthesisArray(n) {
    const result = [];
    
    function backtrack(current, open, close) {
        // Base case
        if (current.length === 2 * n) {
            result.push(current.join(''));
            return;
        }
        
        // Add opening parenthesis
        if (open < n) {
            current.push('(');
            backtrack(current, open + 1, close);
            current.pop(); // Backtrack
        }
        
        // Add closing parenthesis
        if (close < open) {
            current.push(')');
            backtrack(current, open, close + 1);
            current.pop(); // Backtrack
        }
    }
    
    backtrack([], 0, 0);
    return result;
}
```

## Dry Run

**Input:** n = 2

```
Initial: current = "", open = 0, close = 0

1. open < 2, add '('
   current = "(", open = 1, close = 0
   
   1.1. open < 2, add '('
        current = "((", open = 2, close = 0
        
        1.1.1. open = 2, can't add '('
        1.1.2. close < open, add ')'
               current = "(()", open = 2, close = 1
               
               1.1.2.1. open = 2, can't add '('
               1.1.2.2. close < open, add ')'
                        current = "(())", open = 2, close = 2
                        Base case: add to result
                        Backtrack: current = "(()"
               
               Backtrack: current = "(("
        
        Backtrack: current = "("
   
   1.2. close < open, add ')'
        current = "()", open = 1, close = 1
        
        1.2.1. open < 2, add '('
               current = "()(", open = 2, close = 1
               
               1.2.1.1. open = 2, can't add '('
               1.2.1.2. close < open, add ')'
                        current = "()()", open = 2, close = 2
                        Base case: add to result
                        Backtrack: current = "()("
               
               Backtrack: current = "()"
        
        Backtrack: current = "("

Backtrack: current = ""

2. close < open (0 < 0), can't add ')'

Result: ["(())", "()()"]
```

## Complexity Analysis

- **Time Complexity:** O(4^n / √n) - This is the nth Catalan number
- **Space Complexity:** O(n) - Maximum recursion depth is 2n

## Alternative Solutions

### Iterative Approach
```javascript
function generateParenthesisIterative(n) {
    const result = [];
    const queue = [{ current: '', open: 0, close: 0 }];
    
    while (queue.length > 0) {
        const { current, open, close } = queue.shift();
        
        if (current.length === 2 * n) {
            result.push(current);
            continue;
        }
        
        if (open < n) {
            queue.push({ current: current + '(', open: open + 1, close });
        }
        
        if (close < open) {
            queue.push({ current: current + ')', open, close: close + 1 });
        }
    }
    
    return result;
}
```

### Dynamic Programming Approach
```javascript
function generateParenthesisDP(n) {
    const dp = Array(n + 1).fill().map(() => []);
    dp[0] = [''];
    
    for (let i = 1; i <= n; i++) {
        for (let j = 0; j < i; j++) {
            for (const left of dp[j]) {
                for (const right of dp[i - 1 - j]) {
                    dp[i].push('(' + left + ')' + right);
                }
            }
        }
    }
    
    return dp[n];
}
```

## Test Cases

```javascript
// Test cases
console.log(generateParenthesis(1)); // ["()"]
console.log(generateParenthesis(2)); // ["(())", "()()"]
console.log(generateParenthesis(3)); // ["((()))", "(()())", "(())()", "()(())", "()()()"]
console.log(generateParenthesis(0)); // [""]

// Edge cases
console.log(generateParenthesis(4).length); // 14 (4th Catalan number)
```

## Key Insights

1. **Catalan Numbers**: The number of valid parentheses combinations for n pairs is the nth Catalan number
2. **Constraint Satisfaction**: We must maintain open >= close at all times
3. **Backtracking Pattern**: Try all valid choices and backtrack when needed
4. **Pruning**: We can skip invalid branches early by checking constraints
5. **State Management**: Keep track of current string and counts of open/close parentheses

## Related Problems

- [Valid Parentheses](StackQueue/ValidParentheses.md/)
- [Minimum Add to Make Parentheses Valid](StackQueue/MinimumAddToMakeParenthesesValid.md/)
- [Remove Invalid Parentheses](RemoveInvalidParentheses.md/)
- [Longest Valid Parentheses](StackQueue/LongestValidParentheses.md/)
