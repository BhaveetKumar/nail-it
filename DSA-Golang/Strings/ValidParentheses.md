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
```

**Constraints:**
- 1 ≤ s.length ≤ 10⁴
- s consists of parentheses only '()[]{}'

### Explanation

#### **Stack Approach**
- Use a stack to keep track of opening brackets
- When we encounter an opening bracket, push it onto the stack
- When we encounter a closing bracket, check if it matches the top of the stack
- If it matches, pop from stack; otherwise, return false
- At the end, stack should be empty for valid string
- Time Complexity: O(n)
- Space Complexity: O(n)

#### **Counter Approach (for single type)**
- For problems with only one type of parentheses, use a counter
- Increment for opening, decrement for closing
- Counter should never go negative and should be zero at the end
- Time Complexity: O(n)
- Space Complexity: O(1)

### Dry Run

**Input:** `s = "()[]{}"`

| Step | Char | Stack | Action |
|------|------|-------|---------|
| 1 | '(' | ['('] | Push opening bracket |
| 2 | ')' | [] | Match found, pop |
| 3 | '[' | ['['] | Push opening bracket |
| 4 | ']' | [] | Match found, pop |
| 5 | '{' | ['{'] | Push opening bracket |
| 6 | '}' | [] | Match found, pop |

**Result:** `true` (stack is empty)

**Input:** `s = "(]"`

| Step | Char | Stack | Action |
|------|------|-------|---------|
| 1 | '(' | ['('] | Push opening bracket |
| 2 | ']' | ['('] | No match, return false |

**Result:** `false`

### Complexity
- **Time Complexity:** O(n) - Single pass through the string
- **Space Complexity:** O(n) - Stack can contain at most n/2 elements

### Golang Solution

#### **Stack Solution**
```go
func isValid(s string) bool {
    if len(s) == 0 {
        return true
    }
    
    // Create a stack to store opening brackets
    stack := make([]rune, 0)
    
    // Map closing brackets to their corresponding opening brackets
    mapping := map[rune]rune{
        ')': '(',
        '}': '{',
        ']': '[',
    }
    
    // Iterate through each character
    for _, char := range s {
        // If it's an opening bracket, push to stack
        if char == '(' || char == '{' || char == '[' {
            stack = append(stack, char)
        } else {
            // If it's a closing bracket
            if len(stack) == 0 {
                return false // No opening bracket to match
            }
            
            // Check if the top of stack matches the closing bracket
            top := stack[len(stack)-1]
            stack = stack[:len(stack)-1] // Pop from stack
            
            if mapping[char] != top {
                return false // Mismatch
            }
        }
    }
    
    // Stack should be empty for valid string
    return len(stack) == 0
}
```

#### **Optimized Stack Solution**
```go
func isValid(s string) bool {
    if len(s)%2 != 0 {
        return false // Odd length can't be valid
    }
    
    stack := make([]rune, 0, len(s)/2) // Pre-allocate capacity
    
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
        default:
            return false // Invalid character
        }
    }
    
    return len(stack) == 0
}
```

### Alternative Solutions

#### **Using bytes for ASCII optimization**
```go
func isValid(s string) bool {
    if len(s)%2 != 0 {
        return false
    }
    
    stack := make([]byte, 0, len(s)/2)
    
    for i := 0; i < len(s); i++ {
        char := s[i]
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

#### **Counter approach for single type parentheses**
```go
func isValidSingleType(s string) bool {
    count := 0
    
    for _, char := range s {
        if char == '(' {
            count++
        } else if char == ')' {
            count--
            if count < 0 {
                return false // More closing than opening
            }
        }
    }
    
    return count == 0
}
```

### Notes / Variations

#### **Related Problems**
- **Generate Parentheses**: Generate all valid combinations
- **Longest Valid Parentheses**: Find longest valid substring
- **Remove Invalid Parentheses**: Remove minimum number to make valid
- **Valid Parentheses String**: Handle wildcard characters
- **Minimum Add to Make Parentheses Valid**: Minimum additions needed

#### **ICPC Insights**
- **Stack Operations**: Master push/pop operations
- **Early Termination**: Return false as soon as mismatch is found
- **Memory Optimization**: Pre-allocate stack capacity
- **Edge Cases**: Handle empty string and odd length

#### **Go-Specific Optimizations**
```go
// Use rune for Unicode support
for _, char := range s {
    // Process each rune
}

// Use byte for ASCII-only strings
for i := 0; i < len(s); i++ {
    char := s[i]
    // Process each byte
}

// Pre-allocate slice capacity
stack := make([]rune, 0, len(s)/2)
```

#### **Real-World Applications**
- **Code Parsing**: Validate syntax in programming languages
- **Expression Evaluation**: Check mathematical expressions
- **HTML/XML Validation**: Validate nested tags
- **Configuration Files**: Validate nested structures

### Testing

```go
func TestIsValid(t *testing.T) {
    tests := []struct {
        input    string
        expected bool
    }{
        {"()", true},
        {"()[]{}", true},
        {"(]", false},
        {"([)]", false},
        {"{[]}", true},
        {"", true},
        {"(", false},
        {")", false},
        {"((", false},
        {"))", false},
    }
    
    for _, test := range tests {
        result := isValid(test.input)
        if result != test.expected {
            t.Errorf("isValid(%q) = %v, expected %v", 
                test.input, result, test.expected)
        }
    }
}
```

### Visualization

```
Input: "()[]{}"

Step 1: '(' → Stack: ['(']
Step 2: ')' → Match! Stack: []
Step 3: '[' → Stack: ['[']
Step 4: ']' → Match! Stack: []
Step 5: '{' → Stack: ['{']
Step 6: '}' → Match! Stack: []

Result: true (stack is empty)

Input: "(]"

Step 1: '(' → Stack: ['(']
Step 2: ']' → No match! Return false

Result: false
```

### Performance Comparison

| Approach | Time | Space | Pros | Cons |
|----------|------|-------|------|------|
| Stack | O(n) | O(n) | Handles all cases | Uses extra space |
| Counter | O(n) | O(1) | Space efficient | Only for single type |
| Recursive | O(n) | O(n) | Intuitive | Stack overflow risk |

**Recommendation**: Use stack approach for general case, counter for single type parentheses.
