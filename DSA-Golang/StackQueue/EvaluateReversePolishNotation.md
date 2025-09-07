# Evaluate Reverse Polish Notation

### Problem
Evaluate the value of an arithmetic expression in Reverse Polish Notation.

Valid operators are `+`, `-`, `*`, and `/`. Each operand may be an integer or another expression.

Note that division between two integers should truncate toward zero.

It is guaranteed that the given RPN expression is always valid. That means the expression will always evaluate to a result and there will not be any division by zero operation.

**Example:**
```
Input: tokens = ["2","1","+","3","*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9

Input: tokens = ["4","13","5","/","+"]
Output: 6
Explanation: (4 + (13 / 5)) = 6
```

### Golang Solution

```go
import (
    "strconv"
    "strings"
)

func evalRPN(tokens []string) int {
    stack := []int{}
    
    for _, token := range tokens {
        if isOperator(token) {
            if len(stack) < 2 {
                return 0
            }
            
            b := stack[len(stack)-1]
            a := stack[len(stack)-2]
            stack = stack[:len(stack)-2]
            
            result := performOperation(a, b, token)
            stack = append(stack, result)
        } else {
            num, _ := strconv.Atoi(token)
            stack = append(stack, num)
        }
    }
    
    return stack[0]
}

func isOperator(token string) bool {
    return token == "+" || token == "-" || token == "*" || token == "/"
}

func performOperation(a, b int, op string) int {
    switch op {
    case "+":
        return a + b
    case "-":
        return a - b
    case "*":
        return a * b
    case "/":
        return a / b
    default:
        return 0
    }
}
```

### Alternative Solutions

#### **Using Switch Statement**
```go
func evalRPNSwitch(tokens []string) int {
    stack := []int{}
    
    for _, token := range tokens {
        switch token {
        case "+":
            b := stack[len(stack)-1]
            a := stack[len(stack)-2]
            stack = stack[:len(stack)-2]
            stack = append(stack, a+b)
        case "-":
            b := stack[len(stack)-1]
            a := stack[len(stack)-2]
            stack = stack[:len(stack)-2]
            stack = append(stack, a-b)
        case "*":
            b := stack[len(stack)-1]
            a := stack[len(stack)-2]
            stack = stack[:len(stack)-2]
            stack = append(stack, a*b)
        case "/":
            b := stack[len(stack)-1]
            a := stack[len(stack)-2]
            stack = stack[:len(stack)-2]
            stack = append(stack, a/b)
        default:
            num, _ := strconv.Atoi(token)
            stack = append(stack, num)
        }
    }
    
    return stack[0]
}
```

#### **Recursive Approach**
```go
func evalRPNRecursive(tokens []string) int {
    if len(tokens) == 1 {
        num, _ := strconv.Atoi(tokens[0])
        return num
    }
    
    for i, token := range tokens {
        if isOperator(token) {
            left := evalRPNRecursive(tokens[:i])
            right := evalRPNRecursive(tokens[i+1:])
            return performOperation(left, right, token)
        }
    }
    
    return 0
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
