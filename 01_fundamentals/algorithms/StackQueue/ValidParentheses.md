---
# Auto-generated front matter
Title: Validparentheses
LastUpdated: 2025-11-06T20:45:58.702851
Tags: []
Status: draft
---

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
            if top == -1 {
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

#### **Return with Error Info**
```go
type ValidationResult struct {
    IsValid bool
    Error   string
    Position int
    Expected rune
    Found    rune
}

func isValidWithError(s string) ValidationResult {
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
                    Expected: 0,
                    Found:    char,
                }
            }
            
            if stack[len(stack)-1] != pairs[char] {
                return ValidationResult{
                    IsValid: false,
                    Error:   "Mismatched brackets",
                    Position: i,
                    Expected: pairs[char],
                    Found:    char,
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
            Expected: 0,
            Found:    stack[len(stack)-1],
        }
    }
    
    return ValidationResult{IsValid: true}
}
```

#### **Return All Valid Combinations**
```go
func generateValidParentheses(n int) []string {
    var result []string
    
    var backtrack func(string, int, int)
    backtrack = func(current string, open int, close int) {
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

#### **Return with Statistics**
```go
type ParenthesesStats struct {
    IsValid        bool
    TotalBrackets  int
    OpenBrackets   int
    CloseBrackets  int
    MaxDepth       int
    CurrentDepth   int
    BracketCounts  map[rune]int
}

func parenthesesStatistics(s string) ParenthesesStats {
    stack := []rune{}
    pairs := map[rune]rune{
        ')': '(',
        '}': '{',
        ']': '[',
    }
    
    stats := ParenthesesStats{
        BracketCounts: make(map[rune]int),
    }
    
    for _, char := range s {
        stats.BracketCounts[char]++
        
        if char == '(' || char == '{' || char == '[' {
            stack = append(stack, char)
            stats.OpenBrackets++
            stats.CurrentDepth++
            if stats.CurrentDepth > stats.MaxDepth {
                stats.MaxDepth = stats.CurrentDepth
            }
        } else if char == ')' || char == '}' || char == ']' {
            stats.CloseBrackets++
            if len(stack) > 0 {
                stats.CurrentDepth--
            }
            
            if len(stack) == 0 || stack[len(stack)-1] != pairs[char] {
                stats.IsValid = false
                return stats
            }
            stack = stack[:len(stack)-1]
        }
    }
    
    stats.IsValid = len(stack) == 0
    stats.TotalBrackets = stats.OpenBrackets + stats.CloseBrackets
    
    return stats
}
```

#### **Return with Fix Suggestions**
```go
type FixSuggestion struct {
    Position int
    Action   string
    Character rune
}

type ValidationWithFix struct {
    IsValid     bool
    Suggestions []FixSuggestion
}

func isValidWithFix(s string) ValidationWithFix {
    stack := []rune{}
    pairs := map[rune]rune{
        ')': '(',
        '}': '{',
        ']': '[',
    }
    
    var suggestions []FixSuggestion
    
    for i, char := range s {
        if char == '(' || char == '{' || char == '[' {
            stack = append(stack, char)
        } else if char == ')' || char == '}' || char == ']' {
            if len(stack) == 0 {
                suggestions = append(suggestions, FixSuggestion{
                    Position:  i,
                    Action:    "Add opening bracket",
                    Character: pairs[char],
                })
            } else if stack[len(stack)-1] != pairs[char] {
                suggestions = append(suggestions, FixSuggestion{
                    Position:  i,
                    Action:    "Replace with correct closing bracket",
                    Character: pairs[stack[len(stack)-1]],
                })
            } else {
                stack = stack[:len(stack)-1]
            }
        }
    }
    
    // Add suggestions for unmatched opening brackets
    for i := len(stack) - 1; i >= 0; i-- {
        suggestions = append(suggestions, FixSuggestion{
            Position:  len(s),
            Action:    "Add closing bracket",
            Character: pairs[stack[i]],
        })
    }
    
    return ValidationWithFix{
        IsValid:     len(stack) == 0 && len(suggestions) == 0,
        Suggestions: suggestions,
    }
}
```

### Complexity
- **Time Complexity:** O(n) where n is the length of the string
- **Space Complexity:** O(n) for the stack