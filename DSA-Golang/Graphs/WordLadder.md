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

            // Try all possible one-character changes
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

### Complexity

- **Time Complexity:** O(M² × N) where M is word length, N is word list size
- **Space Complexity:** O(M × N)
