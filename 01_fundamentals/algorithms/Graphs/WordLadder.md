---
# Auto-generated front matter
Title: Wordladder
LastUpdated: 2025-11-06T20:45:58.737849
Tags: []
Status: draft
---

# Word Ladder

### Problem
A transformation sequence from word `beginWord` to word `endWord` using a dictionary `wordList` is a sequence of words `beginWord -> s1 -> s2 -> ... -> sk` such that:

- Every adjacent pair of words differs by a single letter.
- Every `si` for `1 <= i <= k` is in `wordList`. Note that `beginWord` does not need to be in `wordList`.
- `sk == endWord`

Given two words, `beginWord` and `endWord`, and a dictionary `wordList`, return the number of words in the shortest transformation sequence from `beginWord` to `endWord`, or `0` if no such sequence exists.

**Example:**
```
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5
Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> "cog", which is 5 words long.
```

### Golang Solution

```go
func ladderLength(beginWord string, endWord string, wordList []string) int {
    wordSet := make(map[string]bool)
    for _, word := range wordList {
        wordSet[word] = true
    }
    
    if !wordSet[endWord] {
        return 0
    }
    
    queue := []string{beginWord}
    level := 1
    
    for len(queue) > 0 {
        size := len(queue)
        
        for i := 0; i < size; i++ {
            word := queue[0]
            queue = queue[1:]
            
            if word == endWord {
                return level
            }
            
            // Try all possible transformations
            for j := 0; j < len(word); j++ {
                for c := 'a'; c <= 'z'; c++ {
                    newWord := word[:j] + string(c) + word[j+1:]
                    
                    if wordSet[newWord] {
                        queue = append(queue, newWord)
                        delete(wordSet, newWord)
                    }
                }
            }
        }
        
        level++
    }
    
    return 0
}
```

### Alternative Solutions

#### **Bidirectional BFS**
```go
func ladderLengthBidirectional(beginWord string, endWord string, wordList []string) int {
    wordSet := make(map[string]bool)
    for _, word := range wordList {
        wordSet[word] = true
    }
    
    if !wordSet[endWord] {
        return 0
    }
    
    beginSet := make(map[string]bool)
    endSet := make(map[string]bool)
    beginSet[beginWord] = true
    endSet[endWord] = true
    
    level := 1
    
    for len(beginSet) > 0 && len(endSet) > 0 {
        if len(beginSet) > len(endSet) {
            beginSet, endSet = endSet, beginSet
        }
        
        newSet := make(map[string]bool)
        
        for word := range beginSet {
            for j := 0; j < len(word); j++ {
                for c := 'a'; c <= 'z'; c++ {
                    newWord := word[:j] + string(c) + word[j+1:]
                    
                    if endSet[newWord] {
                        return level + 1
                    }
                    
                    if wordSet[newWord] {
                        newSet[newWord] = true
                        delete(wordSet, newWord)
                    }
                }
            }
        }
        
        beginSet = newSet
        level++
    }
    
    return 0
}
```

#### **DFS with Memoization**
```go
func ladderLengthDFS(beginWord string, endWord string, wordList []string) int {
    wordSet := make(map[string]bool)
    for _, word := range wordList {
        wordSet[word] = true
    }
    
    if !wordSet[endWord] {
        return 0
    }
    
    memo := make(map[string]int)
    result := dfsLadder(beginWord, endWord, wordSet, memo)
    
    if result == math.MaxInt32 {
        return 0
    }
    
    return result
}

func dfsLadder(current, endWord string, wordSet map[string]bool, memo map[string]int) int {
    if current == endWord {
        return 1
    }
    
    if val, exists := memo[current]; exists {
        return val
    }
    
    minSteps := math.MaxInt32
    
    for j := 0; j < len(current); j++ {
        for c := 'a'; c <= 'z'; c++ {
            newWord := current[:j] + string(c) + current[j+1:]
            
            if wordSet[newWord] {
                steps := dfsLadder(newWord, endWord, wordSet, memo)
                if steps != math.MaxInt32 {
                    minSteps = min(minSteps, steps+1)
                }
            }
        }
    }
    
    memo[current] = minSteps
    return minSteps
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

#### **Return All Paths**
```go
func findLadders(beginWord string, endWord string, wordList []string) [][]string {
    wordSet := make(map[string]bool)
    for _, word := range wordList {
        wordSet[word] = true
    }
    
    if !wordSet[endWord] {
        return [][]string{}
    }
    
    var result [][]string
    queue := [][]string{{beginWord}}
    found := false
    
    for len(queue) > 0 && !found {
        size := len(queue)
        used := make(map[string]bool)
        
        for i := 0; i < size; i++ {
            path := queue[0]
            queue = queue[1:]
            
            word := path[len(path)-1]
            
            for j := 0; j < len(word); j++ {
                for c := 'a'; c <= 'z'; c++ {
                    newWord := word[:j] + string(c) + word[j+1:]
                    
                    if newWord == endWord {
                        newPath := make([]string, len(path))
                        copy(newPath, path)
                        newPath = append(newPath, newWord)
                        result = append(result, newPath)
                        found = true
                    } else if wordSet[newWord] {
                        newPath := make([]string, len(path))
                        copy(newPath, path)
                        newPath = append(newPath, newWord)
                        queue = append(queue, newPath)
                        used[newWord] = true
                    }
                }
            }
        }
        
        // Remove used words from wordSet
        for word := range used {
            delete(wordSet, word)
        }
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(M² × N) where M is the length of each word and N is the total number of words
- **Space Complexity:** O(M × N)