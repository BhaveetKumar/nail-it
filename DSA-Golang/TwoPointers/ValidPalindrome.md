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

#### **Using strings package**
```go
import "strings"

func isPalindromeStrings(s string) bool {
    // Convert to lowercase and remove non-alphanumeric
    var cleaned strings.Builder
    for _, char := range s {
        if (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') || (char >= '0' && char <= '9') {
            cleaned.WriteRune(char)
        }
    }
    
    cleanedStr := strings.ToLower(cleaned.String())
    left, right := 0, len(cleanedStr)-1
    
    for left < right {
        if cleanedStr[left] != cleanedStr[right] {
            return false
        }
        left++
        right--
    }
    
    return true
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
