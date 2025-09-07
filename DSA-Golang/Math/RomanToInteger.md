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

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX.

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
    romanMap := map[byte]int{
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000,
    }
    
    result := 0
    prev := 0
    
    for i := len(s) - 1; i >= 0; i-- {
        current := romanMap[s[i]]
        if current < prev {
            result -= current
        } else {
            result += current
        }
        prev = current
    }
    
    return result
}
```

### Alternative Solutions

#### **Left to Right Approach**
```go
func romanToIntLeftToRight(s string) int {
    romanMap := map[byte]int{
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000,
    }
    
    result := 0
    
    for i := 0; i < len(s); i++ {
        if i+1 < len(s) && romanMap[s[i]] < romanMap[s[i+1]] {
            result -= romanMap[s[i]]
        } else {
            result += romanMap[s[i]]
        }
    }
    
    return result
}
```

#### **Switch Case Approach**
```go
func romanToIntSwitch(s string) int {
    result := 0
    
    for i := 0; i < len(s); i++ {
        switch s[i] {
        case 'I':
            if i+1 < len(s) && (s[i+1] == 'V' || s[i+1] == 'X') {
                result -= 1
            } else {
                result += 1
            }
        case 'V':
            result += 5
        case 'X':
            if i+1 < len(s) && (s[i+1] == 'L' || s[i+1] == 'C') {
                result -= 10
            } else {
                result += 10
            }
        case 'L':
            result += 50
        case 'C':
            if i+1 < len(s) && (s[i+1] == 'D' || s[i+1] == 'M') {
                result -= 100
            } else {
                result += 100
            }
        case 'D':
            result += 500
        case 'M':
            result += 1000
        }
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
