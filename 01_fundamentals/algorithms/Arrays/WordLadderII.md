---
# Auto-generated front matter
Title: Wordladderii
LastUpdated: 2025-11-06T20:45:58.723497
Tags: []
Status: draft
---

# Word Ladder II

### Problem
A transformation sequence from word `beginWord` to word `endWord` using a dictionary `wordList` is a sequence of words `beginWord -> s1 -> s2 -> ... -> sk` such that:

- Every adjacent pair of words differs by a single letter.
- Every `si` for `1 <= i <= k` is in `wordList`. Note that `beginWord` does not need to be in `wordList`.
- `sk == endWord`

Given two words, `beginWord` and `endWord`, and a dictionary `wordList`, return all the shortest transformation sequences from `beginWord` to `endWord`, or an empty list if no such sequences exist.

**Example:**
```
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: [["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]
```

### Golang Solution

```go
func findLadders(beginWord string, endWord string, wordList []string) [][]string {
    wordSet := make(map[string]bool)
    for _, word := range wordList {
        wordSet[word] = true
    }
    
    if !wordSet[endWord] {
        return [][]string{}
    }
    
    // BFS to find shortest distance and build graph
    graph := make(map[string][]string)
    distance := make(map[string]int)
    queue := []string{beginWord}
    distance[beginWord] = 0
    
    for len(queue) > 0 {
        size := len(queue)
        for i := 0; i < size; i++ {
            word := queue[0]
            queue = queue[1:]
            
            for j := 0; j < len(word); j++ {
                for c := 'a'; c <= 'z'; c++ {
                    newWord := word[:j] + string(c) + word[j+1:]
                    
                    if wordSet[newWord] {
                        if _, exists := distance[newWord]; !exists {
                            distance[newWord] = distance[word] + 1
                            queue = append(queue, newWord)
                            graph[word] = append(graph[word], newWord)
                        } else if distance[newWord] == distance[word]+1 {
                            graph[word] = append(graph[word], newWord)
                        }
                    }
                }
            }
        }
    }
    
    // DFS to find all paths
    var result [][]string
    var dfs func(string, []string)
    
    dfs = func(word string, path []string) {
        if word == endWord {
            newPath := make([]string, len(path))
            copy(newPath, path)
            result = append(result, newPath)
            return
        }
        
        for _, nextWord := range graph[word] {
            dfs(nextWord, append(path, nextWord))
        }
    }
    
    dfs(beginWord, []string{beginWord})
    return result
}
```

### Alternative Solutions

#### **Bidirectional BFS with Path Reconstruction**
```go
func findLaddersBidirectional(beginWord string, endWord string, wordList []string) [][]string {
    wordSet := make(map[string]bool)
    for _, word := range wordList {
        wordSet[word] = true
    }
    
    if !wordSet[endWord] {
        return [][]string{}
    }
    
    beginSet := make(map[string]bool)
    endSet := make(map[string]bool)
    beginSet[beginWord] = true
    endSet[endWord] = true
    
    graph := make(map[string][]string)
    found := false
    
    for len(beginSet) > 0 && len(endSet) > 0 && !found {
        if len(beginSet) > len(endSet) {
            beginSet, endSet = endSet, beginSet
        }
        
        newSet := make(map[string]bool)
        
        for word := range beginSet {
            for j := 0; j < len(word); j++ {
                for c := 'a'; c <= 'z'; c++ {
                    newWord := word[:j] + string(c) + word[j+1:]
                    
                    if endSet[newWord] {
                        found = true
                        graph[word] = append(graph[word], newWord)
                    }
                    
                    if wordSet[newWord] {
                        newSet[newWord] = true
                        graph[word] = append(graph[word], newWord)
                    }
                }
            }
        }
        
        for word := range newSet {
            delete(wordSet, word)
        }
        
        beginSet = newSet
    }
    
    var result [][]string
    var dfs func(string, []string)
    
    dfs = func(word string, path []string) {
        if word == endWord {
            newPath := make([]string, len(path))
            copy(newPath, path)
            result = append(result, newPath)
            return
        }
        
        for _, nextWord := range graph[word] {
            dfs(nextWord, append(path, nextWord))
        }
    }
    
    dfs(beginWord, []string{beginWord})
    return result
}
```

### Complexity
- **Time Complexity:** O(N × M²) where N is word list size, M is word length
- **Space Complexity:** O(N × M)
