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
Since an empty string reads the same forward and backward, it is a palindrome.
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

#### **Using String Preprocessing**
```go
import "strings"

func isPalindromePreprocess(s string) bool {
    // Remove non-alphanumeric characters and convert to lowercase
    var cleaned strings.Builder
    
    for _, char := range s {
        if (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') || (char >= '0' && char <= '9') {
            cleaned.WriteRune(char)
        }
    }
    
    cleanedStr := strings.ToLower(cleaned.String())
    
    // Check if palindrome
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

#### **Using Regular Expression**
```go
import (
    "regexp"
    "strings"
)

func isPalindromeRegex(s string) bool {
    // Remove non-alphanumeric characters
    re := regexp.MustCompile(`[^a-zA-Z0-9]`)
    cleaned := re.ReplaceAllString(s, "")
    cleaned = strings.ToLower(cleaned)
    
    // Check if palindrome
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

#### **Return with Details**
```go
type PalindromeResult struct {
    IsPalindrome bool
    Original     string
    Cleaned      string
    Length       int
    Characters   []byte
}

func isPalindromeWithDetails(s string) PalindromeResult {
    var cleaned strings.Builder
    var characters []byte
    
    for _, char := range s {
        if (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') || (char >= '0' && char <= '9') {
            cleaned.WriteRune(char)
            characters = append(characters, byte(char))
        }
    }
    
    cleanedStr := strings.ToLower(cleaned.String())
    
    // Check if palindrome
    isPalindrome := true
    left, right := 0, len(cleanedStr)-1
    for left < right {
        if cleanedStr[left] != cleanedStr[right] {
            isPalindrome = false
            break
        }
        left++
        right--
    }
    
    return PalindromeResult{
        IsPalindrome: isPalindrome,
        Original:     s,
        Cleaned:      cleanedStr,
        Length:       len(cleanedStr),
        Characters:   characters,
    }
}
```

#### **Return All Palindromes**
```go
func findAllPalindromes(s string) []string {
    var palindromes []string
    
    for i := 0; i < len(s); i++ {
        for j := i; j < len(s); j++ {
            substring := s[i : j+1]
            if isPalindrome(substring) {
                palindromes = append(palindromes, substring)
            }
        }
    }
    
    return palindromes
}
```

#### **Return Longest Palindrome**
```go
func longestPalindrome(s string) string {
    if len(s) == 0 {
        return ""
    }
    
    start, end := 0, 0
    
    for i := 0; i < len(s); i++ {
        // Check for odd length palindromes
        len1 := expandAroundCenter(s, i, i)
        // Check for even length palindromes
        len2 := expandAroundCenter(s, i, i+1)
        
        maxLen := max(len1, len2)
        
        if maxLen > end-start {
            start = i - (maxLen-1)/2
            end = i + maxLen/2
        }
    }
    
    return s[start : end+1]
}

func expandAroundCenter(s string, left, right int) int {
    for left >= 0 && right < len(s) && s[left] == s[right] {
        left--
        right++
    }
    return right - left - 1
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

#### **Return with Statistics**
```go
type PalindromeStats struct {
    IsPalindrome    bool
    TotalLength     int
    CleanedLength   int
    AlphanumericCount int
    NonAlphanumericCount int
    CharacterFreq   map[byte]int
    LongestPalindrome string
    AllPalindromes   []string
}

func palindromeStatistics(s string) PalindromeStats {
    var cleaned strings.Builder
    alphanumericCount := 0
    characterFreq := make(map[byte]int)
    
    for _, char := range s {
        if (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') || (char >= '0' && char <= '9') {
            cleaned.WriteRune(char)
            alphanumericCount++
            characterFreq[byte(char)]++
        }
    }
    
    cleanedStr := strings.ToLower(cleaned.String())
    
    // Check if palindrome
    isPalindrome := true
    left, right := 0, len(cleanedStr)-1
    for left < right {
        if cleanedStr[left] != cleanedStr[right] {
            isPalindrome = false
            break
        }
        left++
        right--
    }
    
    longestPalindrome := longestPalindrome(s)
    allPalindromes := findAllPalindromes(s)
    
    return PalindromeStats{
        IsPalindrome:        isPalindrome,
        TotalLength:         len(s),
        CleanedLength:       len(cleanedStr),
        AlphanumericCount:   alphanumericCount,
        NonAlphanumericCount: len(s) - alphanumericCount,
        CharacterFreq:       characterFreq,
        LongestPalindrome:   longestPalindrome,
        AllPalindromes:      allPalindromes,
    }
}
```

### Complexity
- **Time Complexity:** O(n) where n is the length of the string
- **Space Complexity:** O(1) for in-place, O(n) for preprocessing