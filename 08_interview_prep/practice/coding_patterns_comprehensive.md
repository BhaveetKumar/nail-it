# Coding Patterns Comprehensive

Essential coding patterns for technical interviews.

## 🎯 Two Pointers Pattern

### Basic Two Pointers
```go
// Two Sum (Sorted Array)
func twoSumSorted(nums []int, target int) []int {
    left, right := 0, len(nums)-1
    
    for left < right {
        sum := nums[left] + nums[right]
        if sum == target {
            return []int{left, right}
        } else if sum < target {
            left++
        } else {
            right--
        }
    }
    
    return []int{-1, -1}
}

// Remove Duplicates
func removeDuplicates(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    
    slow := 0
    for fast := 1; fast < len(nums); fast++ {
        if nums[fast] != nums[slow] {
            slow++
            nums[slow] = nums[fast]
        }
    }
    
    return slow + 1
}
```

### Sliding Window
```go
// Longest Substring Without Repeating Characters
func lengthOfLongestSubstring(s string) int {
    charMap := make(map[byte]int)
    left, maxLen := 0, 0
    
    for right := 0; right < len(s); right++ {
        if pos, exists := charMap[s[right]]; exists && pos >= left {
            left = pos + 1
        }
        
        charMap[s[right]] = right
        maxLen = max(maxLen, right-left+1)
    }
    
    return maxLen
}

// Minimum Window Substring
func minWindow(s string, t string) string {
    if len(s) < len(t) {
        return ""
    }
    
    need := make(map[byte]int)
    for i := range t {
        need[t[i]]++
    }
    
    left, right := 0, 0
    valid := 0
    window := make(map[byte]int)
    start, length := 0, math.MaxInt32
    
    for right < len(s) {
        c := s[right]
        right++
        
        if need[c] > 0 {
            window[c]++
            if window[c] == need[c] {
                valid++
            }
        }
        
        for valid == len(need) {
            if right-left < length {
                start = left
                length = right - left
            }
            
            d := s[left]
            left++
            
            if need[d] > 0 {
                if window[d] == need[d] {
                    valid--
                }
                window[d]--
            }
        }
    }
    
    if length == math.MaxInt32 {
        return ""
    }
    
    return s[start : start+length]
}
```

## 🔄 Dynamic Programming

### Basic DP Patterns
```go
// Climbing Stairs
func climbStairs(n int) int {
    if n <= 2 {
        return n
    }
    
    dp := make([]int, n+1)
    dp[1] = 1
    dp[2] = 2
    
    for i := 3; i <= n; i++ {
        dp[i] = dp[i-1] + dp[i-2]
    }
    
    return dp[n]
}

// House Robber
func rob(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    if len(nums) == 1 {
        return nums[0]
    }
    
    prev2 := nums[0]
    prev1 := max(nums[0], nums[1])
    
    for i := 2; i < len(nums); i++ {
        current := max(prev1, prev2+nums[i])
        prev2 = prev1
        prev1 = current
    }
    
    return prev1
}
```

### 2D DP
```go
// Longest Common Subsequence
func longestCommonSubsequence(text1 string, text2 string) int {
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
```

## 🌳 Tree Algorithms

### DFS Patterns
```go
// Number of Islands
func numIslands(grid [][]byte) int {
    if len(grid) == 0 {
        return 0
    }
    
    m, n := len(grid), len(grid[0])
    count := 0
    
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if grid[i][j] == '1' {
                dfs(grid, i, j, m, n)
                count++
            }
        }
    }
    
    return count
}

func dfs(grid [][]byte, i, j, m, n int) {
    if i < 0 || i >= m || j < 0 || j >= n || grid[i][j] != '1' {
        return
    }
    
    grid[i][j] = '0'
    dfs(grid, i+1, j, m, n)
    dfs(grid, i-1, j, m, n)
    dfs(grid, i, j+1, m, n)
    dfs(grid, i, j-1, m, n)
}
```

### BFS Patterns
```go
// Word Ladder
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

## 🔧 Data Structures

### LRU Cache
```go
type LRUCache struct {
    capacity int
    cache    map[int]*Node
    head     *Node
    tail     *Node
}

type Node struct {
    key   int
    value int
    prev  *Node
    next  *Node
}

func NewLRUCache(capacity int) *LRUCache {
    head := &Node{key: 0, value: 0}
    tail := &Node{key: 0, value: 0}
    head.next = tail
    tail.prev = head
    
    return &LRUCache{
        capacity: capacity,
        cache:    make(map[int]*Node),
        head:     head,
        tail:     tail,
    }
}

func (lru *LRUCache) Get(key int) int {
    if node, exists := lru.cache[key]; exists {
        lru.moveToHead(node)
        return node.value
    }
    return -1
}

func (lru *LRUCache) Put(key int, value int) {
    if node, exists := lru.cache[key]; exists {
        node.value = value
        lru.moveToHead(node)
    } else {
        newNode := &Node{key: key, value: value}
        lru.cache[key] = newNode
        lru.addToHead(newNode)
        
        if len(lru.cache) > lru.capacity {
            tail := lru.removeTail()
            delete(lru.cache, tail.key)
        }
    }
}
```

### Trie
```go
type Trie struct {
    children map[byte]*Trie
    isEnd    bool
}

func NewTrie() *Trie {
    return &Trie{
        children: make(map[byte]*Trie),
        isEnd:    false,
    }
}

func (t *Trie) Insert(word string) {
    node := t
    for i := 0; i < len(word); i++ {
        char := word[i]
        if node.children[char] == nil {
            node.children[char] = NewTrie()
        }
        node = node.children[char]
    }
    node.isEnd = true
}

func (t *Trie) Search(word string) bool {
    node := t
    for i := 0; i < len(word); i++ {
        char := word[i]
        if node.children[char] == nil {
            return false
        }
        node = node.children[char]
    }
    return node.isEnd
}
```

## 🎯 System Design Coding

### Rate Limiter
```go
type TokenBucket struct {
    capacity     int
    tokens       int
    lastRefill   time.Time
    refillRate   int
    mutex        sync.Mutex
}

func NewTokenBucket(capacity, refillRate int) *TokenBucket {
    return &TokenBucket{
        capacity:   capacity,
        tokens:     capacity,
        lastRefill: time.Now(),
        refillRate: refillRate,
    }
}

func (tb *TokenBucket) Allow() bool {
    tb.mutex.Lock()
    defer tb.mutex.Unlock()
    
    now := time.Now()
    tokensToAdd := int(now.Sub(tb.lastRefill).Seconds()) * tb.refillRate
    tb.tokens = min(tb.capacity, tb.tokens+tokensToAdd)
    tb.lastRefill = now
    
    if tb.tokens > 0 {
        tb.tokens--
        return true
    }
    
    return false
}
```

### Message Queue
```go
type MessageQueue struct {
    subscribers map[string][]chan Message
    mutex       sync.RWMutex
}

type Message struct {
    Topic   string
    Content interface{}
    ID      string
}

func NewMessageQueue() *MessageQueue {
    return &MessageQueue{
        subscribers: make(map[string][]chan Message),
    }
}

func (mq *MessageQueue) Subscribe(topic string) <-chan Message {
    mq.mutex.Lock()
    defer mq.mutex.Unlock()
    
    ch := make(chan Message, 100)
    mq.subscribers[topic] = append(mq.subscribers[topic], ch)
    
    return ch
}

func (mq *MessageQueue) Publish(topic string, content interface{}) {
    mq.mutex.RLock()
    subscribers := mq.subscribers[topic]
    mq.mutex.RUnlock()
    
    message := Message{
        Topic:   topic,
        Content: content,
        ID:      generateMessageID(),
    }
    
    for _, ch := range subscribers {
        select {
        case ch <- message:
        default:
            // Channel is full, skip this subscriber
        }
    }
}
```

## 🎯 Interview Tips

### Common Patterns
1. **Two Pointers**: Sorted arrays, palindromes
2. **Sliding Window**: Subarray problems
3. **Hash Map**: Frequency counting, lookups
4. **Stack**: Matching brackets, monotonic
5. **Queue**: BFS, level-order traversal

### Time Complexity
- **O(1)**: Constant time
- **O(log n)**: Binary search
- **O(n)**: Linear scan
- **O(n log n)**: Sorting
- **O(n²)**: Nested loops
- **O(2ⁿ)**: Exponential

### Best Practices
1. **Clarify Requirements**: Ask questions
2. **Think Out Loud**: Explain process
3. **Start Simple**: Brute force first
4. **Test Examples**: Walk through code
5. **Handle Edge Cases**: Empty inputs, etc.

---

**Last Updated**: December 2024  
**Category**: Coding Patterns Comprehensive  
**Complexity**: Senior Level
