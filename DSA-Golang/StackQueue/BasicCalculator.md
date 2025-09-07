# Basic Calculator

### Problem
Given a string `s` representing a valid expression, implement a basic calculator to evaluate it, and return the result of the evaluation.

Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as `eval()`.

**Example:**
```
Input: s = "1 + 1"
Output: 2

Input: s = " 2-1 + 2 "
Output: 3

Input: s = "(1+(4+5+2)-3)+(6+8)"
Output: 23
```

### Golang Solution

```go
func calculate(s string) int {
    stack := []int{}
    result := 0
    number := 0
    sign := 1
    
    for i := 0; i < len(s); i++ {
        char := s[i]
        
        if char >= '0' && char <= '9' {
            number = number*10 + int(char-'0')
        } else if char == '+' {
            result += sign * number
            number = 0
            sign = 1
        } else if char == '-' {
            result += sign * number
            number = 0
            sign = -1
        } else if char == '(' {
            stack = append(stack, result)
            stack = append(stack, sign)
            result = 0
            sign = 1
        } else if char == ')' {
            result += sign * number
            number = 0
            result *= stack[len(stack)-1] // sign
            stack = stack[:len(stack)-1]
            result += stack[len(stack)-1] // previous result
            stack = stack[:len(stack)-1]
        }
    }
    
    return result + sign*number
}
```

### Alternative Solutions

#### **Using Two Stacks**
```go
func calculateTwoStacks(s string) int {
    numbers := []int{}
    operators := []byte{}
    
    i := 0
    for i < len(s) {
        char := s[i]
        
        if char == ' ' {
            i++
            continue
        }
        
        if char >= '0' && char <= '9' {
            num := 0
            for i < len(s) && s[i] >= '0' && s[i] <= '9' {
                num = num*10 + int(s[i]-'0')
                i++
            }
            numbers = append(numbers, num)
            continue
        }
        
        if char == '(' {
            operators = append(operators, char)
        } else if char == ')' {
            for len(operators) > 0 && operators[len(operators)-1] != '(' {
                numbers = append(numbers, applyOperator(numbers, operators))
            }
            operators = operators[:len(operators)-1] // remove '('
        } else if char == '+' || char == '-' {
            for len(operators) > 0 && operators[len(operators)-1] != '(' {
                numbers = append(numbers, applyOperator(numbers, operators))
            }
            operators = append(operators, char)
        }
        
        i++
    }
    
    for len(operators) > 0 {
        numbers = append(numbers, applyOperator(numbers, operators))
    }
    
    return numbers[0]
}

func applyOperator(numbers []int, operators []byte) int {
    if len(numbers) < 2 || len(operators) == 0 {
        return 0
    }
    
    b := numbers[len(numbers)-1]
    a := numbers[len(numbers)-2]
    op := operators[len(operators)-1]
    
    numbers = numbers[:len(numbers)-2]
    operators = operators[:len(operators)-1]
    
    if op == '+' {
        return a + b
    } else if op == '-' {
        return a - b
    }
    
    return 0
}
```

#### **Recursive Approach**
```go
func calculateRecursive(s string) int {
    i := 0
    return calculateHelper(s, &i)
}

func calculateHelper(s string, i *int) int {
    result := 0
    number := 0
    sign := 1
    
    for *i < len(s) {
        char := s[*i]
        
        if char >= '0' && char <= '9' {
            number = number*10 + int(char-'0')
        } else if char == '+' {
            result += sign * number
            number = 0
            sign = 1
        } else if char == '-' {
            result += sign * number
            number = 0
            sign = -1
        } else if char == '(' {
            *i++
            result += sign * calculateHelper(s, i)
        } else if char == ')' {
            result += sign * number
            return result
        }
        
        *i++
    }
    
    return result + sign*number
}
```

#### **Shunting Yard Algorithm**
```go
func calculateShuntingYard(s string) int {
    // Convert infix to postfix using Shunting Yard algorithm
    postfix := infixToPostfix(s)
    return evaluatePostfix(postfix)
}

func infixToPostfix(s string) []string {
    var output []string
    var operators []byte
    precedence := map[byte]int{'+': 1, '-': 1, '(': 0}
    
    i := 0
    for i < len(s) {
        char := s[i]
        
        if char == ' ' {
            i++
            continue
        }
        
        if char >= '0' && char <= '9' {
            num := ""
            for i < len(s) && s[i] >= '0' && s[i] <= '9' {
                num += string(s[i])
                i++
            }
            output = append(output, num)
            continue
        }
        
        if char == '(' {
            operators = append(operators, char)
        } else if char == ')' {
            for len(operators) > 0 && operators[len(operators)-1] != '(' {
                output = append(output, string(operators[len(operators)-1]))
                operators = operators[:len(operators)-1]
            }
            operators = operators[:len(operators)-1] // remove '('
        } else if char == '+' || char == '-' {
            for len(operators) > 0 && precedence[operators[len(operators)-1]] >= precedence[char] {
                output = append(output, string(operators[len(operators)-1]))
                operators = operators[:len(operators)-1]
            }
            operators = append(operators, char)
        }
        
        i++
    }
    
    for len(operators) > 0 {
        output = append(output, string(operators[len(operators)-1]))
        operators = operators[:len(operators)-1]
    }
    
    return output
}

func evaluatePostfix(tokens []string) int {
    var stack []int
    
    for _, token := range tokens {
        if token == "+" || token == "-" {
            if len(stack) < 2 {
                continue
            }
            b := stack[len(stack)-1]
            a := stack[len(stack)-2]
            stack = stack[:len(stack)-2]
            
            if token == "+" {
                stack = append(stack, a+b)
            } else {
                stack = append(stack, a-b)
            }
        } else {
            num, _ := strconv.Atoi(token)
            stack = append(stack, num)
        }
    }
    
    if len(stack) > 0 {
        return stack[0]
    }
    return 0
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
