# Strings Pattern

> **Master string manipulation and pattern matching with Go implementations**

## 📋 Problems

### **Pattern Matching**
- [Valid Parentheses](./ValidParentheses.md) - Stack-based validation
- [Longest Substring Without Repeating Characters](./LongestSubstringWithoutRepeatingCharacters.md) - Sliding window
- [Longest Palindromic Substring](./LongestPalindromicSubstring.md) - Expand around centers
- [Regular Expression Matching](./RegularExpressionMatching.md) - DP with wildcards
- [Wildcard Matching](./WildcardMatching.md) - Advanced pattern matching

### **String Manipulation**
- [String to Integer (atoi)](./StringToInteger.md) - Parsing with edge cases
- [Roman to Integer](./RomanToInteger.md) - Symbol to value conversion
- [Integer to Roman](./IntegerToRoman.md) - Value to symbol conversion
- [Valid Anagram](./ValidAnagram.md) - Character frequency comparison
- [Group Anagrams](./GroupAnagrams.md) - Anagram grouping

### **Advanced String Algorithms**
- [Longest Common Prefix](./LongestCommonPrefix.md) - Prefix comparison
- [Implement strStr()](./ImplementStrStr.md) - KMP algorithm
- [Minimum Window Substring](./MinimumWindowSubstring.md) - Sliding window
- [Permutation in String](./PermutationInString.md) - Sliding window with frequency
- [Find All Anagrams in a String](./FindAllAnagramsInString.md) - Sliding window

---

## 🎯 Key Concepts

### **String Operations in Go**
- **String Immutability**: Strings are immutable in Go
- **Rune Handling**: Use `[]rune` for Unicode characters
- **String Building**: Use `strings.Builder` for efficient concatenation
- **Regular Expressions**: Use `regexp` package for pattern matching

### **Common Patterns**
- **Two Pointers**: For palindromes and string comparison
- **Sliding Window**: For substring problems
- **Hash Map**: For character frequency counting
- **Stack**: For nested structures (parentheses, brackets)

### **Performance Tips**
- **String Builder**: Use for multiple concatenations
- **Pre-allocate Slices**: When size is known
- **Avoid String Concatenation**: Use slice operations instead

---

## 🛠️ Go-Specific Tips

### **String Manipulation**
```go
// Convert string to rune slice for Unicode handling
s := "hello世界"
runes := []rune(s)

// Efficient string building
var builder strings.Builder
builder.WriteString("hello")
builder.WriteString(" world")
result := builder.String()

// String to byte slice (for ASCII)
bytes := []byte("hello")

// Byte slice to string
str := string(bytes)
```

### **Character Frequency Counting**
```go
// Using map for frequency counting
func countFreq(s string) map[rune]int {
    freq := make(map[rune]int)
    for _, char := range s {
        freq[char]++
    }
    return freq
}

// Using array for ASCII characters
func countFreqASCII(s string) [256]int {
    var freq [256]int
    for i := 0; i < len(s); i++ {
        freq[s[i]]++
    }
    return freq
}
```

### **String Comparison**
```go
// Case-insensitive comparison
func equalsIgnoreCase(s1, s2 string) bool {
    return strings.ToLower(s1) == strings.ToLower(s2)
}

// Custom comparison function
func compareStrings(s1, s2 string) int {
    return strings.Compare(s1, s2)
}
```

### **Regular Expressions**
```go
import "regexp"

// Compile regex pattern
pattern := regexp.MustCompile(`\d+`)

// Find all matches
matches := pattern.FindAllString("abc123def456", -1)

// Replace matches
result := pattern.ReplaceAllString("abc123def456", "X")
```

---

## 🎯 Interview Tips

### **How to Identify String Problems**
1. **Pattern Matching**: Use regex or custom algorithms
2. **Substring Problems**: Use sliding window technique
3. **Character Frequency**: Use hash map or array
4. **Palindrome Problems**: Use two pointers
5. **String Transformation**: Use DP or greedy approach

### **Common String Problem Patterns**
- **Validation**: Check if string meets certain criteria
- **Parsing**: Extract information from formatted strings
- **Transformation**: Convert between different formats
- **Search**: Find patterns or substrings
- **Comparison**: Compare strings with different criteria

### **Optimization Tips**
- **Use appropriate data structures**: Map for frequency, array for ASCII
- **Avoid unnecessary conversions**: Work with bytes when possible
- **Pre-allocate memory**: Use make() with known capacity
- **Use built-in functions**: Leverage Go's string package
