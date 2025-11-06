---
# Auto-generated front matter
Title: Longestsubstringwithatmostkdistinctcharacters
LastUpdated: 2025-11-06T20:45:58.711590
Tags: []
Status: draft
---

# Longest Substring with At Most K Distinct Characters

### Problem
Given a string `s` and an integer `k`, return the length of the longest substring of `s` that contains at most `k` distinct characters.

**Example:**
```
Input: s = "eceba", k = 2
Output: 3
Explanation: The substring is "ece" with length 3.

Input: s = "aa", k = 1
Output: 2
Explanation: The substring is "aa" with length 2.
```

### Golang Solution

```go
func lengthOfLongestSubstringKDistinct(s string, k int) int {
    if k == 0 {
        return 0
    }
    
    charCount := make(map[byte]int)
    left := 0
    maxLength := 0
    
    for right := 0; right < len(s); right++ {
        charCount[s[right]]++
        
        for len(charCount) > k {
            charCount[s[left]]--
            if charCount[s[left]] == 0 {
                delete(charCount, s[left])
            }
            left++
        }
        
        maxLength = max(maxLength, right-left+1)
    }
    
    return maxLength
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

### Alternative Solutions

#### **Using Array for Character Count**
```go
func lengthOfLongestSubstringKDistinctArray(s string, k int) int {
    if k == 0 {
        return 0
    }
    
    charCount := make([]int, 256)
    distinctCount := 0
    left := 0
    maxLength := 0
    
    for right := 0; right < len(s); right++ {
        if charCount[s[right]] == 0 {
            distinctCount++
        }
        charCount[s[right]]++
        
        for distinctCount > k {
            charCount[s[left]]--
            if charCount[s[left]] == 0 {
                distinctCount--
            }
            left++
        }
        
        maxLength = max(maxLength, right-left+1)
    }
    
    return maxLength
}
```

#### **Return with Substring**
```go
type SubstringResult struct {
    Length    int
    Substring string
    Start     int
    End       int
}

func lengthOfLongestSubstringKDistinctWithSubstring(s string, k int) SubstringResult {
    if k == 0 {
        return SubstringResult{Length: 0, Substring: "", Start: 0, End: -1}
    }
    
    charCount := make(map[byte]int)
    left := 0
    maxLength := 0
    bestStart := 0
    bestEnd := 0
    
    for right := 0; right < len(s); right++ {
        charCount[s[right]]++
        
        for len(charCount) > k {
            charCount[s[left]]--
            if charCount[s[left]] == 0 {
                delete(charCount, s[left])
            }
            left++
        }
        
        if right-left+1 > maxLength {
            maxLength = right - left + 1
            bestStart = left
            bestEnd = right
        }
    }
    
    return SubstringResult{
        Length:    maxLength,
        Substring: s[bestStart : bestEnd+1],
        Start:     bestStart,
        End:       bestEnd,
    }
}
```

#### **Return All Valid Substrings**
```go
func allValidSubstrings(s string, k int) []SubstringResult {
    var results []SubstringResult
    
    for i := 0; i < len(s); i++ {
        charCount := make(map[byte]int)
        distinctCount := 0
        
        for j := i; j < len(s); j++ {
            if charCount[s[j]] == 0 {
                distinctCount++
            }
            charCount[s[j]]++
            
            if distinctCount <= k {
                results = append(results, SubstringResult{
                    Length:    j - i + 1,
                    Substring: s[i : j+1],
                    Start:     i,
                    End:       j,
                })
            } else {
                break
            }
        }
    }
    
    return results
}
```

#### **Return with Character Info**
```go
type CharacterInfo struct {
    Character byte
    Count     int
    Positions []int
}

type SubstringWithChars struct {
    Length      int
    Substring   string
    Start       int
    End         int
    Characters  []CharacterInfo
    DistinctCount int
}

func lengthOfLongestSubstringKDistinctWithChars(s string, k int) SubstringWithChars {
    if k == 0 {
        return SubstringWithChars{Length: 0, Substring: "", Start: 0, End: -1}
    }
    
    charCount := make(map[byte]int)
    charPositions := make(map[byte][]int)
    left := 0
    maxLength := 0
    bestStart := 0
    bestEnd := 0
    
    for right := 0; right < len(s); right++ {
        charCount[s[right]]++
        charPositions[s[right]] = append(charPositions[s[right]], right)
        
        for len(charCount) > k {
            charCount[s[left]]--
            if charCount[s[left]] == 0 {
                delete(charCount, s[left])
                delete(charPositions, s[left])
            }
            left++
        }
        
        if right-left+1 > maxLength {
            maxLength = right - left + 1
            bestStart = left
            bestEnd = right
        }
    }
    
    var characters []CharacterInfo
    for char, count := range charCount {
        characters = append(characters, CharacterInfo{
            Character: char,
            Count:     count,
            Positions: charPositions[char],
        })
    }
    
    return SubstringWithChars{
        Length:        maxLength,
        Substring:     s[bestStart : bestEnd+1],
        Start:         bestStart,
        End:           bestEnd,
        Characters:    characters,
        DistinctCount: len(charCount),
    }
}
```

#### **Return Statistics**
```go
type SubstringStats struct {
    MaxLength      int
    MinLength      int
    AvgLength      float64
    TotalSubstrings int
    DistinctChars   int
    MostFrequentChar byte
    CharFrequency   map[byte]int
}

func substringStatistics(s string, k int) SubstringStats {
    if k == 0 {
        return SubstringStats{CharFrequency: make(map[byte]int)}
    }
    
    var lengths []int
    charFreq := make(map[byte]int)
    
    for i := 0; i < len(s); i++ {
        charCount := make(map[byte]int)
        distinctCount := 0
        
        for j := i; j < len(s); j++ {
            if charCount[s[j]] == 0 {
                distinctCount++
            }
            charCount[s[j]]++
            charFreq[s[j]]++
            
            if distinctCount <= k {
                lengths = append(lengths, j-i+1)
            } else {
                break
            }
        }
    }
    
    if len(lengths) == 0 {
        return SubstringStats{CharFrequency: charFreq}
    }
    
    maxLength := lengths[0]
    minLength := lengths[0]
    sum := 0
    
    for _, length := range lengths {
        if length > maxLength {
            maxLength = length
        }
        if length < minLength {
            minLength = length
        }
        sum += length
    }
    
    mostFrequent := byte(0)
    maxFreq := 0
    for char, freq := range charFreq {
        if freq > maxFreq {
            maxFreq = freq
            mostFrequent = char
        }
    }
    
    return SubstringStats{
        MaxLength:       maxLength,
        MinLength:       minLength,
        AvgLength:       float64(sum) / float64(len(lengths)),
        TotalSubstrings: len(lengths),
        DistinctChars:   len(charFreq),
        MostFrequentChar: mostFrequent,
        CharFrequency:   charFreq,
    }
}
```

### Complexity
- **Time Complexity:** O(n) where n is the length of the string
- **Space Complexity:** O(k) for character count map