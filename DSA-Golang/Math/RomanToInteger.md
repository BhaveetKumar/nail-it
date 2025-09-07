# Roman to Integer

### Problem
Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

```
Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
```

For example, 2 is written as II in Roman numeral, just two ones added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

- I can be placed before V (5) and X (10) to make 4 and 9.
- X can be placed before L (50) and C (100) to make 40 and 90.
- C can be placed before D (500) and M (1000) to make 400 and 900.

Given a roman numeral, convert it to an integer.

**Example:**
```
Input: s = "III"
Output: 3

Input: s = "LVIII"
Output: 58
Explanation: L = 50, V= 5, III = 3.

Input: s = "MCMXC"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90.
```

### Golang Solution

```go
func romanToInt(s string) int {
    romanValues := map[byte]int{
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000,
    }
    
    result := 0
    prevValue := 0
    
    for i := len(s) - 1; i >= 0; i-- {
        currentValue := romanValues[s[i]]
        
        if currentValue < prevValue {
            result -= currentValue
        } else {
            result += currentValue
        }
        
        prevValue = currentValue
    }
    
    return result
}
```

### Alternative Solutions

#### **Left to Right Processing**
```go
func romanToIntLeftToRight(s string) int {
    romanValues := map[byte]int{
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000,
    }
    
    result := 0
    
    for i := 0; i < len(s); i++ {
        currentValue := romanValues[s[i]]
        
        if i+1 < len(s) {
            nextValue := romanValues[s[i+1]]
            if currentValue < nextValue {
                result -= currentValue
            } else {
                result += currentValue
            }
        } else {
            result += currentValue
        }
    }
    
    return result
}
```

#### **Using Switch Statement**
```go
func romanToIntSwitch(s string) int {
    result := 0
    prevValue := 0
    
    for i := len(s) - 1; i >= 0; i-- {
        var currentValue int
        
        switch s[i] {
        case 'I':
            currentValue = 1
        case 'V':
            currentValue = 5
        case 'X':
            currentValue = 10
        case 'L':
            currentValue = 50
        case 'C':
            currentValue = 100
        case 'D':
            currentValue = 500
        case 'M':
            currentValue = 1000
        }
        
        if currentValue < prevValue {
            result -= currentValue
        } else {
            result += currentValue
        }
        
        prevValue = currentValue
    }
    
    return result
}
```

#### **Using Array for ASCII Values**
```go
func romanToIntArray(s string) int {
    values := make([]int, 128)
    values['I'] = 1
    values['V'] = 5
    values['X'] = 10
    values['L'] = 50
    values['C'] = 100
    values['D'] = 500
    values['M'] = 1000
    
    result := 0
    prevValue := 0
    
    for i := len(s) - 1; i >= 0; i-- {
        currentValue := values[s[i]]
        
        if currentValue < prevValue {
            result -= currentValue
        } else {
            result += currentValue
        }
        
        prevValue = currentValue
    }
    
    return result
}
```

#### **Return with Validation**
```go
type RomanResult struct {
    Value int
    Valid bool
    Error string
}

func romanToIntWithValidation(s string) RomanResult {
    if len(s) == 0 {
        return RomanResult{Valid: false, Error: "Empty string"}
    }
    
    romanValues := map[byte]int{
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000,
    }
    
    result := 0
    prevValue := 0
    
    for i := len(s) - 1; i >= 0; i-- {
        if val, exists := romanValues[s[i]]; exists {
            currentValue := val
            
            if currentValue < prevValue {
                result -= currentValue
            } else {
                result += currentValue
            }
            
            prevValue = currentValue
        } else {
            return RomanResult{Valid: false, Error: "Invalid character: " + string(s[i])}
        }
    }
    
    return RomanResult{Value: result, Valid: true}
}
```

#### **Integer to Roman**
```go
func intToRoman(num int) string {
    values := []int{1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1}
    symbols := []string{"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"}
    
    result := ""
    
    for i := 0; i < len(values); i++ {
        count := num / values[i]
        for j := 0; j < count; j++ {
            result += symbols[i]
        }
        num %= values[i]
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n) where n is the length of the string
- **Space Complexity:** O(1)