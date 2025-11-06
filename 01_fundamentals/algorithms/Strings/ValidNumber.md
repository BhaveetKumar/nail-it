---
# Auto-generated front matter
Title: Validnumber
LastUpdated: 2025-11-06T20:45:58.689064
Tags: []
Status: draft
---

# Valid Number

### Problem
A valid number can be split up into these components (in order):

1. A decimal number or an integer.
2. (Optional) An 'e' or 'E', followed by an integer.

A decimal number can be split up into these components (in order):

1. (Optional) A sign character (either '+' or '-').
2. One of the following formats:
   - One or more digits, followed by a dot '.'.
   - One or more digits, followed by a dot '.', followed by one or more digits.
   - A dot '.', followed by one or more digits.

An integer can be split up into these components (in order):

1. (Optional) A sign character (either '+' or '-').
2. One or more digits.

**Example:**
```
Input: s = "0"
Output: true

Input: s = "e"
Output: false

Input: s = "."
Output: false
```

### Golang Solution

```go
func isNumber(s string) bool {
    s = strings.TrimSpace(s)
    if len(s) == 0 {
        return false
    }
    
    seenDigit := false
    seenDot := false
    seenE := false
    
    for i, char := range s {
        switch char {
        case '+', '-':
            if i > 0 && s[i-1] != 'e' && s[i-1] != 'E' {
                return false
            }
        case '.':
            if seenDot || seenE {
                return false
            }
            seenDot = true
        case 'e', 'E':
            if seenE || !seenDigit {
                return false
            }
            seenE = true
            seenDigit = false // Reset for the part after 'e'
        default:
            if char < '0' || char > '9' {
                return false
            }
            seenDigit = true
        }
    }
    
    return seenDigit
}
```

### Alternative Solutions

#### **State Machine Approach**
```go
func isNumberStateMachine(s string) bool {
    s = strings.TrimSpace(s)
    if len(s) == 0 {
        return false
    }
    
    state := 0
    for _, char := range s {
        switch state {
        case 0: // Initial state
            if char == '+' || char == '-' {
                state = 1
            } else if char >= '0' && char <= '9' {
                state = 2
            } else if char == '.' {
                state = 3
            } else {
                return false
            }
        case 1: // Sign seen
            if char >= '0' && char <= '9' {
                state = 2
            } else if char == '.' {
                state = 3
            } else {
                return false
            }
        case 2: // Digit seen
            if char >= '0' && char <= '9' {
                state = 2
            } else if char == '.' {
                state = 4
            } else if char == 'e' || char == 'E' {
                state = 5
            } else {
                return false
            }
        case 3: // Dot seen (no digits before)
            if char >= '0' && char <= '9' {
                state = 4
            } else {
                return false
            }
        case 4: // Digit after dot
            if char >= '0' && char <= '9' {
                state = 4
            } else if char == 'e' || char == 'E' {
                state = 5
            } else {
                return false
            }
        case 5: // E seen
            if char == '+' || char == '-' {
                state = 6
            } else if char >= '0' && char <= '9' {
                state = 7
            } else {
                return false
            }
        case 6: // Sign after E
            if char >= '0' && char <= '9' {
                state = 7
            } else {
                return false
            }
        case 7: // Digit after E
            if char >= '0' && char <= '9' {
                state = 7
            } else {
                return false
            }
        }
    }
    
    return state == 2 || state == 4 || state == 7
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
