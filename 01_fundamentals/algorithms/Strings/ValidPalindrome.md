# Valid Palindrome

### Problem
A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

Given a string `s`, return `true` if it is a palindrome, or `false` otherwise.

**Example:**
```
Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.

Input: s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.

Input: s = " "
Output: true
Explanation: s is an empty string "" after removing non-alphanumeric characters.
```

### Golang Solution

```go
func isPalindrome(s string) bool {
    left, right := 0, len(s)-1
    
    for left < right {
        // Skip non-alphanumeric characters from left
        for left < right && !isAlphanumeric(s[left]) {
            left++
        }
        
        // Skip non-alphanumeric characters from right
        for left < right && !isAlphanumeric(s[right]) {
            right--
        }
        
        // Compare characters (case insensitive)
        if toLowerCase(s[left]) != toLowerCase(s[right]) {
            return false
        }
        
        left++
        right--
    }
    
    return true
}

func isAlphanumeric(c byte) bool {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
}

func toLowerCase(c byte) byte {
    if c >= 'A' && c <= 'Z' {
        return c + 32
    }
    return c
}
```

### Alternative Solutions

#### **Using Built-in Functions**
```go
import (
    "regexp"
    "strings"
)

func isPalindromeBuiltin(s string) bool {
    // Remove non-alphanumeric characters and convert to lowercase
    re := regexp.MustCompile(`[^a-zA-Z0-9]`)
    cleaned := strings.ToLower(re.ReplaceAllString(s, ""))
    
    left, right := 0, len(cleaned)-1
    
    for left < right {
        if cleaned[left] != cleaned[right] {
            return false
        }
        left++
        right--
    }
    
    return true
}
```

#### **Recursive Approach**
```go
func isPalindromeRecursive(s string) bool {
    cleaned := cleanString(s)
    return isPalindromeHelper(cleaned, 0, len(cleaned)-1)
}

func cleanString(s string) string {
    var result strings.Builder
    
    for _, char := range s {
        if (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') || (char >= '0' && char <= '9') {
            if char >= 'A' && char <= 'Z' {
                result.WriteRune(char + 32)
            } else {
                result.WriteRune(char)
            }
        }
    }
    
    return result.String()
}

func isPalindromeHelper(s string, left, right int) bool {
    if left >= right {
        return true
    }
    
    if s[left] != s[right] {
        return false
    }
    
    return isPalindromeHelper(s, left+1, right-1)
}
```

#### **Using Stack**
```go
func isPalindromeStack(s string) bool {
    var stack []byte
    var cleaned []byte
    
    // Clean the string
    for i := 0; i < len(s); i++ {
        if isAlphanumeric(s[i]) {
            cleaned = append(cleaned, toLowerCase(s[i]))
        }
    }
    
    // Push first half to stack
    for i := 0; i < len(cleaned)/2; i++ {
        stack = append(stack, cleaned[i])
    }
    
    // Compare second half with stack
    start := len(cleaned) / 2
    if len(cleaned)%2 == 1 {
        start++ // Skip middle character for odd length
    }
    
    for i := start; i < len(cleaned); i++ {
        if len(stack) == 0 || stack[len(stack)-1] != cleaned[i] {
            return false
        }
        stack = stack[:len(stack)-1]
    }
    
    return true
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for two pointers, O(n) for recursive/stack
