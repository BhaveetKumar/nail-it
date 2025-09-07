# Valid Parentheses

### Problem
Given a string `s` containing just the characters `'('`, `')'`, `'{'`, `'}'`, `'['` and `']'`, determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.

**Example:**
```
Input: s = "()"
Output: true

Input: s = "()[]{}"
Output: true

Input: s = "(]"
Output: false

Input: s = "([)]"
Output: false
```

### Golang Solution

```go
func isValid(s string) bool {
    stack := []rune{}
    pairs := map[rune]rune{
        ')': '(',
        '}': '{',
        ']': '[',
    }
    
    for _, char := range s {
        if char == '(' || char == '{' || char == '[' {
            stack = append(stack, char)
        } else if char == ')' || char == '}' || char == ']' {
            if len(stack) == 0 || stack[len(stack)-1] != pairs[char] {
                return false
            }
            stack = stack[:len(stack)-1]
        }
    }
    
    return len(stack) == 0
}
```

### Alternative Solutions

#### **Using Switch Statement**
```go
func isValidSwitch(s string) bool {
    stack := []rune{}
    
    for _, char := range s {
        switch char {
        case '(', '{', '[':
            stack = append(stack, char)
        case ')':
            if len(stack) == 0 || stack[len(stack)-1] != '(' {
                return false
            }
            stack = stack[:len(stack)-1]
        case '}':
            if len(stack) == 0 || stack[len(stack)-1] != '{' {
                return false
            }
            stack = stack[:len(stack)-1]
        case ']':
            if len(stack) == 0 || stack[len(stack)-1] != '[' {
                return false
            }
            stack = stack[:len(stack)-1]
        }
    }
    
    return len(stack) == 0
}
```

#### **Using Array as Stack**
```go
func isValidArray(s string) bool {
    stack := make([]byte, len(s))
    top := -1
    
    for i := 0; i < len(s); i++ {
        char := s[i]
        
        if char == '(' || char == '{' || char == '[' {
            top++
            stack[top] = char
        } else if char == ')' || char == '}' || char == ']' {
            if top < 0 {
                return false
            }
            
            var expected byte
            switch char {
            case ')':
                expected = '('
            case '}':
                expected = '{'
            case ']':
                expected = '['
            }
            
            if stack[top] != expected {
                return false
            }
            top--
        }
    }
    
    return top == -1
}
```

#### **Using Counter (Limited Use Case)**
```go
func isValidCounter(s string) bool {
    if len(s)%2 != 0 {
        return false
    }
    
    count := 0
    for _, char := range s {
        if char == '(' || char == '{' || char == '[' {
            count++
        } else {
            count--
        }
        if count < 0 {
            return false
        }
    }
    
    return count == 0
}
```

#### **Recursive Approach**
```go
func isValidRecursive(s string) bool {
    if len(s) == 0 {
        return true
    }
    
    if len(s) == 1 {
        return false
    }
    
    // Find matching pair
    for i := 0; i < len(s)-1; i++ {
        if isMatching(s[i], s[i+1]) {
            return isValidRecursive(s[:i] + s[i+2:])
        }
    }
    
    return false
}

func isMatching(open, close byte) bool {
    return (open == '(' && close == ')') ||
           (open == '{' && close == '}') ||
           (open == '[' && close == ']')
}
```

#### **Return with Error Details**
```go
type ValidationResult struct {
    IsValid bool
    Error   string
    Position int
}

func isValidWithDetails(s string) ValidationResult {
    stack := []rune{}
    pairs := map[rune]rune{
        ')': '(',
        '}': '{',
        ']': '[',
    }
    
    for i, char := range s {
        if char == '(' || char == '{' || char == '[' {
            stack = append(stack, char)
        } else if char == ')' || char == '}' || char == ']' {
            if len(stack) == 0 {
                return ValidationResult{
                    IsValid: false,
                    Error:   "Unmatched closing bracket",
                    Position: i,
                }
            }
            
            if stack[len(stack)-1] != pairs[char] {
                return ValidationResult{
                    IsValid: false,
                    Error:   "Mismatched brackets",
                    Position: i,
                }
            }
            
            stack = stack[:len(stack)-1]
        }
    }
    
    if len(stack) > 0 {
        return ValidationResult{
            IsValid: false,
            Error:   "Unmatched opening brackets",
            Position: len(s),
        }
    }
    
    return ValidationResult{IsValid: true}
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)