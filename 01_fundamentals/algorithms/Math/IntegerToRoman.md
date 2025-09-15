# Integer to Roman

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

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX.

Given an integer, convert it to a roman numeral.

**Example:**
```
Input: num = 3
Output: "III"

Input: num = 58
Output: "LVIII"

Input: num = 1994
Output: "MCMXC"
```

### Golang Solution

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

### Alternative Solutions

#### **Switch Case Approach**
```go
func intToRomanSwitch(num int) string {
    result := ""
    
    // Handle thousands
    for num >= 1000 {
        result += "M"
        num -= 1000
    }
    
    // Handle hundreds
    if num >= 900 {
        result += "CM"
        num -= 900
    } else if num >= 500 {
        result += "D"
        num -= 500
        for num >= 100 {
            result += "C"
            num -= 100
        }
    } else if num >= 400 {
        result += "CD"
        num -= 400
    } else {
        for num >= 100 {
            result += "C"
            num -= 100
        }
    }
    
    // Handle tens
    if num >= 90 {
        result += "XC"
        num -= 90
    } else if num >= 50 {
        result += "L"
        num -= 50
        for num >= 10 {
            result += "X"
            num -= 10
        }
    } else if num >= 40 {
        result += "XL"
        num -= 40
    } else {
        for num >= 10 {
            result += "X"
            num -= 10
        }
    }
    
    // Handle ones
    if num >= 9 {
        result += "IX"
        num -= 9
    } else if num >= 5 {
        result += "V"
        num -= 5
        for num >= 1 {
            result += "I"
            num -= 1
        }
    } else if num >= 4 {
        result += "IV"
        num -= 4
    } else {
        for num >= 1 {
            result += "I"
            num -= 1
        }
    }
    
    return result
}
```

#### **Recursive Approach**
```go
func intToRomanRecursive(num int) string {
    if num == 0 {
        return ""
    }
    
    if num >= 1000 {
        return "M" + intToRomanRecursive(num-1000)
    } else if num >= 900 {
        return "CM" + intToRomanRecursive(num-900)
    } else if num >= 500 {
        return "D" + intToRomanRecursive(num-500)
    } else if num >= 400 {
        return "CD" + intToRomanRecursive(num-400)
    } else if num >= 100 {
        return "C" + intToRomanRecursive(num-100)
    } else if num >= 90 {
        return "XC" + intToRomanRecursive(num-90)
    } else if num >= 50 {
        return "L" + intToRomanRecursive(num-50)
    } else if num >= 40 {
        return "XL" + intToRomanRecursive(num-40)
    } else if num >= 10 {
        return "X" + intToRomanRecursive(num-10)
    } else if num >= 9 {
        return "IX" + intToRomanRecursive(num-9)
    } else if num >= 5 {
        return "V" + intToRomanRecursive(num-5)
    } else if num >= 4 {
        return "IV" + intToRomanRecursive(num-4)
    } else {
        return "I" + intToRomanRecursive(num-1)
    }
}
```

#### **Using StringBuilder**
```go
import "strings"

func intToRomanStringBuilder(num int) string {
    values := []int{1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1}
    symbols := []string{"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"}
    
    var result strings.Builder
    
    for i := 0; i < len(values); i++ {
        count := num / values[i]
        for j := 0; j < count; j++ {
            result.WriteString(symbols[i])
        }
        num %= values[i]
    }
    
    return result.String()
}
```

### Complexity
- **Time Complexity:** O(1) - constant number of operations
- **Space Complexity:** O(1)
