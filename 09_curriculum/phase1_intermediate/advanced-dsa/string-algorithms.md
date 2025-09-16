# String Algorithms

## Overview

This module covers advanced string algorithms including pattern matching, string processing, and text analysis algorithms. These algorithms are essential for text processing, bioinformatics, and data analysis applications.

## Table of Contents

1. [Pattern Matching](#pattern-matching/)
2. [String Processing](#string-processing/)
3. [Suffix Arrays](#suffix-arrays/)
4. [Trie Applications](#trie-applications/)
5. [String Hashing](#string-hashing/)
6. [Applications](#applications/)
7. [Complexity Analysis](#complexity-analysis/)
8. [Follow-up Questions](#follow-up-questions/)

## Pattern Matching

### KMP (Knuth-Morris-Pratt) Algorithm

The KMP algorithm efficiently finds all occurrences of a pattern in a text using the failure function (LPS array).

#### Theory

- Preprocesses the pattern to create a failure function
- Uses the failure function to avoid unnecessary comparisons
- Time complexity: O(m + n) where m is pattern length, n is text length

#### Implementations

##### Golang Implementation

```go
package main

import "fmt"

func computeLPS(pattern string) []int {
    m := len(pattern)
    lps := make([]int, m)
    length := 0
    i := 1
    
    for i < m {
        if pattern[i] == pattern[length] {
            length++
            lps[i] = length
            i++
        } else {
            if length != 0 {
                length = lps[length-1]
            } else {
                lps[i] = 0
                i++
            }
        }
    }
    
    return lps
}

func kmpSearch(text, pattern string) []int {
    n, m := len(text), len(pattern)
    if m == 0 {
        return []int{}
    }
    
    lps := computeLPS(pattern)
    result := []int{}
    
    i, j := 0, 0 // i for text, j for pattern
    
    for i < n {
        if text[i] == pattern[j] {
            i++
            j++
        }
        
        if j == m {
            result = append(result, i-j)
            j = lps[j-1]
        } else if i < n && text[i] != pattern[j] {
            if j != 0 {
                j = lps[j-1]
            } else {
                i++
            }
        }
    }
    
    return result
}

func main() {
    text := "ABABDABACDABABCABAB"
    pattern := "ABABCABAB"
    
    result := kmpSearch(text, pattern)
    fmt.Printf("Pattern found at indices: %v\n", result)
    
    // Count occurrences
    fmt.Printf("Number of occurrences: %d\n", len(result))
}
```

##### Node.js Implementation

```javascript
function computeLPS(pattern) {
    const m = pattern.length;
    const lps = Array(m).fill(0);
    let length = 0;
    let i = 1;
    
    while (i < m) {
        if (pattern[i] === pattern[length]) {
            length++;
            lps[i] = length;
            i++;
        } else {
            if (length !== 0) {
                length = lps[length - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
    
    return lps;
}

function kmpSearch(text, pattern) {
    const n = text.length;
    const m = pattern.length;
    if (m === 0) return [];
    
    const lps = computeLPS(pattern);
    const result = [];
    
    let i = 0, j = 0; // i for text, j for pattern
    
    while (i < n) {
        if (text[i] === pattern[j]) {
            i++;
            j++;
        }
        
        if (j === m) {
            result.push(i - j);
            j = lps[j - 1];
        } else if (i < n && text[i] !== pattern[j]) {
            if (j !== 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
    }
    
    return result;
}

// Example usage
const text = "ABABDABACDABABCABAB";
const pattern = "ABABCABAB";

const result = kmpSearch(text, pattern);
console.log(`Pattern found at indices: ${result}`);
console.log(`Number of occurrences: ${result.length}`);
```

### Rabin-Karp Algorithm

The Rabin-Karp algorithm uses rolling hash to find pattern matches in text.

#### Theory

- Uses polynomial rolling hash function
- Compares hash values before doing character-by-character comparison
- Average time complexity: O(m + n), worst case: O(mn)

#### Implementations

##### Golang Implementation

```go
package main

import "fmt"

func rabinKarpSearch(text, pattern string) []int {
    n, m := len(text), len(pattern)
    if m == 0 || m > n {
        return []int{}
    }
    
    const base = 256
    const mod = 101
    
    // Calculate hash of pattern
    patternHash := 0
    for i := 0; i < m; i++ {
        patternHash = (patternHash*base + int(pattern[i])) % mod
    }
    
    // Calculate hash of first window
    textHash := 0
    for i := 0; i < m; i++ {
        textHash = (textHash*base + int(text[i])) % mod
    }
    
    result := []int{}
    
    // Check first window
    if patternHash == textHash && text[:m] == pattern {
        result = append(result, 0)
    }
    
    // Calculate base^(m-1) for rolling hash
    h := 1
    for i := 0; i < m-1; i++ {
        h = (h * base) % mod
    }
    
    // Rolling hash
    for i := 1; i <= n-m; i++ {
        // Remove leading digit, add trailing digit
        textHash = (textHash - int(text[i-1])*h + mod) % mod
        textHash = (textHash*base + int(text[i+m-1])) % mod
        
        if patternHash == textHash && text[i:i+m] == pattern {
            result = append(result, i)
        }
    }
    
    return result
}

func main() {
    text := "ABABDABACDABABCABAB"
    pattern := "ABABCABAB"
    
    result := rabinKarpSearch(text, pattern)
    fmt.Printf("Pattern found at indices: %v\n", result)
}
```

## String Processing

### Longest Common Subsequence (LCS)

#### Problem
Find the length of the longest subsequence present in both strings.

#### Implementations

##### Golang Implementation

```go
package main

import "fmt"

func longestCommonSubsequence(text1, text2 string) int {
    m, n := len(text1), len(text2)
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }
    
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if text1[i-1] == text2[j-1] {
                dp[i][j] = dp[i-1][j-1] + 1
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            }
        }
    }
    
    return dp[m][n]
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

func main() {
    text1 := "abcde"
    text2 := "ace"
    result := longestCommonSubsequence(text1, text2)
    fmt.Printf("LCS length: %d\n", result)
}
```

### Edit Distance (Levenshtein Distance)

#### Problem
Find the minimum number of operations (insert, delete, replace) to convert one string to another.

#### Implementations

##### Golang Implementation

```go
package main

import "fmt"

func minDistance(word1, word2 string) int {
    m, n := len(word1), len(word2)
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }
    
    // Initialize base cases
    for i := 0; i <= m; i++ {
        dp[i][0] = i
    }
    for j := 0; j <= n; j++ {
        dp[0][j] = j
    }
    
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if word1[i-1] == word2[j-1] {
                dp[i][j] = dp[i-1][j-1]
            } else {
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            }
        }
    }
    
    return dp[m][n]
}

func min(a, b, c int) int {
    if a < b && a < c {
        return a
    } else if b < c {
        return b
    }
    return c
}

func main() {
    word1 := "horse"
    word2 := "ros"
    result := minDistance(word1, word2)
    fmt.Printf("Edit distance: %d\n", result)
}
```

## Suffix Arrays

### Theory

Suffix arrays are data structures that store all suffixes of a string in sorted order, enabling efficient string operations.

#### Implementations

##### Golang Implementation

```go
package main

import (
    "fmt"
    "sort"
)

type Suffix struct {
    index int
    suffix string
}

type SuffixArray struct {
    text string
    suffixes []Suffix
}

func NewSuffixArray(text string) *SuffixArray {
    sa := &SuffixArray{
        text: text,
        suffixes: make([]Suffix, len(text)),
    }
    
    for i := 0; i < len(text); i++ {
        sa.suffixes[i] = Suffix{
            index: i,
            suffix: text[i:],
        }
    }
    
    sort.Slice(sa.suffixes, func(i, j int) bool {
        return sa.suffixes[i].suffix < sa.suffixes[j].suffix
    })
    
    return sa
}

func (sa *SuffixArray) Search(pattern string) []int {
    result := []int{}
    n := len(sa.suffixes)
    
    // Binary search for pattern
    left, right := 0, n-1
    start := -1
    
    // Find first occurrence
    for left <= right {
        mid := (left + right) / 2
        if sa.suffixes[mid].suffix >= pattern {
            if sa.suffixes[mid].suffix[:min(len(sa.suffixes[mid].suffix), len(pattern))] == pattern {
                start = mid
            }
            right = mid - 1
        } else {
            left = mid + 1
        }
    }
    
    if start == -1 {
        return result
    }
    
    // Find all occurrences
    for i := start; i < n; i++ {
        if sa.suffixes[i].suffix[:min(len(sa.suffixes[i].suffix), len(pattern))] == pattern {
            result = append(result, sa.suffixes[i].index)
        } else {
            break
        }
    }
    
    return result
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

func main() {
    text := "banana"
    sa := NewSuffixArray(text)
    
    fmt.Println("Suffix Array:")
    for i, suffix := range sa.suffixes {
        fmt.Printf("%d: %s (index %d)\n", i, suffix.suffix, suffix.index)
    }
    
    pattern := "an"
    result := sa.Search(pattern)
    fmt.Printf("Pattern '%s' found at indices: %v\n", pattern, result)
}
```

## Trie Applications

### Theory

Tries (prefix trees) are tree-like data structures used for efficient string operations, especially prefix-based searches.

#### Implementations

##### Golang Implementation

```go
package main

import "fmt"

type TrieNode struct {
    children map[rune]*TrieNode
    isEnd    bool
}

type Trie struct {
    root *TrieNode
}

func NewTrie() *Trie {
    return &Trie{
        root: &TrieNode{
            children: make(map[rune]*TrieNode),
            isEnd:    false,
        },
    }
}

func (t *Trie) Insert(word string) {
    node := t.root
    for _, char := range word {
        if node.children[char] == nil {
            node.children[char] = &TrieNode{
                children: make(map[rune]*TrieNode),
                isEnd:    false,
            }
        }
        node = node.children[char]
    }
    node.isEnd = true
}

func (t *Trie) Search(word string) bool {
    node := t.root
    for _, char := range word {
        if node.children[char] == nil {
            return false
        }
        node = node.children[char]
    }
    return node.isEnd
}

func (t *Trie) StartsWith(prefix string) bool {
    node := t.root
    for _, char := range prefix {
        if node.children[char] == nil {
            return false
        }
        node = node.children[char]
    }
    return true
}

func (t *Trie) GetAllWordsWithPrefix(prefix string) []string {
    node := t.root
    for _, char := range prefix {
        if node.children[char] == nil {
            return []string{}
        }
        node = node.children[char]
    }
    
    var result []string
    t.dfs(node, prefix, &result)
    return result
}

func (t *Trie) dfs(node *TrieNode, current string, result *[]string) {
    if node.isEnd {
        *result = append(*result, current)
    }
    
    for char, child := range node.children {
        t.dfs(child, current+string(char), result)
    }
}

func main() {
    trie := NewTrie()
    
    words := []string{"apple", "app", "application", "apply", "banana", "band"}
    for _, word := range words {
        trie.Insert(word)
    }
    
    fmt.Printf("Search 'app': %t\n", trie.Search("app"))
    fmt.Printf("Search 'apps': %t\n", trie.Search("apps"))
    fmt.Printf("StartsWith 'app': %t\n", trie.StartsWith("app"))
    
    wordsWithPrefix := trie.GetAllWordsWithPrefix("app")
    fmt.Printf("Words with prefix 'app': %v\n", wordsWithPrefix)
}
```

## String Hashing

### Theory

String hashing uses hash functions to quickly compare strings and enable efficient string operations.

#### Implementations

##### Golang Implementation

```go
package main

import "fmt"

type StringHasher struct {
    base int
    mod  int
    power []int
}

func NewStringHasher(base, mod int) *StringHasher {
    return &StringHasher{
        base: base,
        mod:  mod,
    }
}

func (sh *StringHasher) precomputePowers(maxLen int) {
    sh.power = make([]int, maxLen+1)
    sh.power[0] = 1
    for i := 1; i <= maxLen; i++ {
        sh.power[i] = (sh.power[i-1] * sh.base) % sh.mod
    }
}

func (sh *StringHasher) computeHash(s string) int {
    hash := 0
    for _, char := range s {
        hash = (hash*sh.base + int(char)) % sh.mod
    }
    return hash
}

func (sh *StringHasher) computeRollingHash(s string) []int {
    n := len(s)
    hash := make([]int, n+1)
    
    for i := 0; i < n; i++ {
        hash[i+1] = (hash[i]*sh.base + int(s[i])) % sh.mod
    }
    
    return hash
}

func (sh *StringHasher) getSubstringHash(hash []int, l, r int) int {
    if l == 0 {
        return hash[r]
    }
    return (hash[r] - hash[l]*sh.power[r-l]%sh.mod + sh.mod) % sh.mod
}

func main() {
    hasher := NewStringHasher(256, 1000000007)
    hasher.precomputePowers(1000)
    
    text := "hello world"
    hash := hasher.computeRollingHash(text)
    
    fmt.Printf("Text: %s\n", text)
    fmt.Printf("Hash of 'hello': %d\n", hasher.getSubstringHash(hash, 0, 5))
    fmt.Printf("Hash of 'world': %d\n", hasher.getSubstringHash(hash, 6, 11))
}
```

## Follow-up Questions

### 1. Pattern Matching
**Q: When would you use KMP vs Rabin-Karp for pattern matching?**
A: Use KMP when you need guaranteed O(m+n) time complexity and the pattern is relatively short. Use Rabin-Karp when you need to find multiple patterns or when the pattern is very long, as it can be parallelized and has better average-case performance.

### 2. String Processing
**Q: How do you optimize space complexity in LCS and edit distance problems?**
A: Use rolling arrays or space-optimized DP where you only keep the previous row/column. For LCS, you can reduce space from O(mn) to O(min(m,n)) by swapping arrays.

### 3. Trie Applications
**Q: What are the trade-offs between using tries vs hash tables for string operations?**
A: Tries provide prefix-based operations and ordered traversal but use more space. Hash tables are faster for exact matches but don't support prefix operations. Choose based on your specific use case.

## Complexity Analysis

| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| KMP | O(m + n) | O(m) | Pattern preprocessing |
| Rabin-Karp | O(m + n) avg | O(1) | Rolling hash |
| LCS | O(mn) | O(mn) | DP approach |
| Edit Distance | O(mn) | O(mn) | DP approach |
| Suffix Array | O(n log n) | O(n) | Construction |
| Trie Operations | O(k) | O(ALPHABET_SIZE * N * k) | k = string length |

## Applications

1. **Pattern Matching**: Text search, DNA sequence analysis
2. **String Processing**: Spell checkers, diff algorithms
3. **Suffix Arrays**: String compression, bioinformatics
4. **Tries**: Autocomplete, IP routing, spell checkers
5. **String Hashing**: Fast string comparison, rolling hash

---

**Next**: [Mathematical Algorithms](mathematical-algorithms.md/) | **Previous**: [Advanced DSA](README.md/) | **Up**: [Advanced DSA](README.md/)
