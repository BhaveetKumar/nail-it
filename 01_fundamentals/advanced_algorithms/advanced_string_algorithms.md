# Advanced String Algorithms

## Table of Contents
- [Introduction](#introduction/)
- [String Matching Algorithms](#string-matching-algorithms/)
- [Suffix Arrays and Trees](#suffix-arrays-and-trees/)
- [String Compression](#string-compression/)
- [Pattern Matching](#pattern-matching/)
- [String Processing](#string-processing/)
- [Advanced Applications](#advanced-applications/)

## Introduction

Advanced string algorithms provide efficient solutions for complex string processing problems, pattern matching, and text analysis.

## String Matching Algorithms

### KMP (Knuth-Morris-Pratt) Algorithm

**Problem**: Find all occurrences of a pattern in a text efficiently.

```go
// KMP Algorithm Implementation
func KMP(text, pattern string) []int {
    n, m := len(text), len(pattern)
    if m == 0 {
        return []int{}
    }
    
    // Build failure function (LPS array)
    lps := buildLPS(pattern)
    
    var result []int
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

func buildLPS(pattern string) []int {
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
```

### Rabin-Karp Algorithm

**Problem**: Find pattern in text using rolling hash.

```go
// Rabin-Karp Algorithm with Rolling Hash
func RabinKarp(text, pattern string) []int {
    n, m := len(text), len(pattern)
    if m == 0 || m > n {
        return []int{}
    }
    
    const base = 256
    const mod = 1000000007
    
    // Calculate hash of pattern
    patternHash := 0
    for i := 0; i < m; i++ {
        patternHash = (patternHash*base + int(pattern[i])) % mod
    }
    
    // Calculate hash of first window
    windowHash := 0
    power := 1
    for i := 0; i < m; i++ {
        windowHash = (windowHash*base + int(text[i])) % mod
        if i < m-1 {
            power = (power * base) % mod
        }
    }
    
    var result []int
    
    // Slide the window
    for i := 0; i <= n-m; i++ {
        if windowHash == patternHash {
            if text[i:i+m] == pattern {
                result = append(result, i)
            }
        }
        
        if i < n-m {
            // Remove leading digit, add trailing digit
            windowHash = (windowHash - int(text[i])*power) % mod
            windowHash = (windowHash + mod) % mod
            windowHash = (windowHash*base + int(text[i+m])) % mod
        }
    }
    
    return result
}
```

### Z-Algorithm

**Problem**: Find all occurrences of pattern in text using Z-array.

```go
// Z-Algorithm Implementation
func ZAlgorithm(text, pattern string) []int {
    concat := pattern + "$" + text
    z := buildZArray(concat)
    
    var result []int
    patternLen := len(pattern)
    
    for i := patternLen + 1; i < len(z); i++ {
        if z[i] == patternLen {
            result = append(result, i-patternLen-1)
        }
    }
    
    return result
}

func buildZArray(s string) []int {
    n := len(s)
    z := make([]int, n)
    
    l, r := 0, 0
    
    for i := 1; i < n; i++ {
        if i <= r {
            z[i] = min(r-i+1, z[i-l])
        }
        
        for i+z[i] < n && s[z[i]] == s[i+z[i]] {
            z[i]++
        }
        
        if i+z[i]-1 > r {
            l = i
            r = i + z[i] - 1
        }
    }
    
    return z
}
```

## Suffix Arrays and Trees

### Suffix Array Construction

**Problem**: Build suffix array for efficient string operations.

```go
// Suffix Array Implementation
type SuffixArray struct {
    text string
    sa   []int
    lcp  []int
}

func NewSuffixArray(text string) *SuffixArray {
    sa := &SuffixArray{
        text: text,
        sa:   make([]int, len(text)),
    }
    
    sa.buildSuffixArray()
    sa.buildLCPArray()
    
    return sa
}

func (sa *SuffixArray) buildSuffixArray() {
    n := len(sa.text)
    
    // Initialize with single characters
    for i := 0; i < n; i++ {
        sa.sa[i] = i
    }
    
    // Sort by first character
    sort.Slice(sa.sa, func(i, j int) bool {
        return sa.text[sa.sa[i]] < sa.text[sa.sa[j]]
    })
    
    // Sort by increasing lengths
    for k := 1; k < n; k *= 2 {
        sa.sortByLength(k)
    }
}

func (sa *SuffixArray) sortByLength(k int) {
    n := len(sa.text)
    
    // Create equivalence classes
    c := make([]int, n)
    c[sa.sa[0]] = 0
    
    for i := 1; i < n; i++ {
        if sa.text[sa.sa[i]] == sa.text[sa.sa[i-1]] {
            c[sa.sa[i]] = c[sa.sa[i-1]]
        } else {
            c[sa.sa[i]] = c[sa.sa[i-1]] + 1
        }
    }
    
    // Sort by pairs
    sort.Slice(sa.sa, func(i, j int) bool {
        if c[sa.sa[i]] != c[sa.sa[j]] {
            return c[sa.sa[i]] < c[sa.sa[j]]
        }
        
        // Compare second part
        a := sa.sa[i] + k
        b := sa.sa[j] + k
        
        if a >= n || b >= n {
            return a > b
        }
        
        return c[a] < c[b]
    })
}

func (sa *SuffixArray) buildLCPArray() {
    n := len(sa.text)
    sa.lcp = make([]int, n)
    
    // Build inverse suffix array
    inv := make([]int, n)
    for i := 0; i < n; i++ {
        inv[sa.sa[i]] = i
    }
    
    k := 0
    for i := 0; i < n; i++ {
        if inv[i] == n-1 {
            k = 0
            continue
        }
        
        j := sa.sa[inv[i]+1]
        
        for i+k < n && j+k < n && sa.text[i+k] == sa.text[j+k] {
            k++
        }
        
        sa.lcp[inv[i]] = k
        
        if k > 0 {
            k--
        }
    }
}

func (sa *SuffixArray) Search(pattern string) []int {
    n := len(sa.text)
    m := len(pattern)
    
    // Binary search for pattern
    left := 0
    right := n - 1
    
    for left <= right {
        mid := (left + right) / 2
        suffix := sa.sa[mid]
        
        // Compare pattern with suffix
        cmp := sa.compare(suffix, pattern)
        
        if cmp == 0 {
            // Found match, find all occurrences
            return sa.findAllOccurrences(mid, pattern)
        } else if cmp < 0 {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return []int{}
}

func (sa *SuffixArray) compare(suffix int, pattern string) int {
    n := len(sa.text)
    m := len(pattern)
    
    for i := 0; i < m; i++ {
        if suffix+i >= n {
            return -1
        }
        
        if sa.text[suffix+i] < pattern[i] {
            return -1
        } else if sa.text[suffix+i] > pattern[i] {
            return 1
        }
    }
    
    return 0
}

func (sa *SuffixArray) findAllOccurrences(mid int, pattern string) []int {
    n := len(sa.text)
    m := len(pattern)
    
    var result []int
    
    // Find left boundary
    left := mid
    for left > 0 && sa.compare(sa.sa[left-1], pattern) == 0 {
        left--
    }
    
    // Find right boundary
    right := mid
    for right < n-1 && sa.compare(sa.sa[right+1], pattern) == 0 {
        right++
    }
    
    // Collect all occurrences
    for i := left; i <= right; i++ {
        result = append(result, sa.sa[i])
    }
    
    return result
}
```

### Suffix Tree Implementation

**Problem**: Build suffix tree for efficient string operations.

```go
// Suffix Tree Implementation
type SuffixTree struct {
    root *Node
    text string
}

type Node struct {
    children map[byte]*Node
    start    int
    end      int
    suffixLink *Node
    isLeaf   bool
}

func NewSuffixTree(text string) *SuffixTree {
    st := &SuffixTree{
        root: &Node{
            children: make(map[byte]*Node),
            start:    -1,
            end:      -1,
        },
        text: text + "$", // Add sentinel
    }
    
    st.buildSuffixTree()
    return st
}

func (st *SuffixTree) buildSuffixTree() {
    n := len(st.text)
    activeNode := st.root
    activeEdge := byte(0)
    activeLength := 0
    remainingSuffixCount := 0
    lastNewNode := (*Node)(nil)
    
    for i := 0; i < n; i++ {
        lastNewNode = nil
        remainingSuffixCount++
        
        for remainingSuffixCount > 0 {
            if activeLength == 0 {
                activeEdge = st.text[i]
            }
            
            if activeNode.children[activeEdge] == nil {
                // Create new leaf node
                activeNode.children[activeEdge] = &Node{
                    children: make(map[byte]*Node),
                    start:    i,
                    end:      n - 1,
                    isLeaf:   true,
                }
                
                if lastNewNode != nil {
                    lastNewNode.suffixLink = activeNode
                    lastNewNode = nil
                }
            } else {
                nextNode := activeNode.children[activeEdge]
                edgeLength := nextNode.end - nextNode.start + 1
                
                if activeLength >= edgeLength {
                    activeLength -= edgeLength
                    activeNode = nextNode
                    activeEdge = st.text[i-activeLength]
                    continue
                }
                
                if st.text[nextNode.start+activeLength] == st.text[i] {
                    if lastNewNode != nil && activeNode != st.root {
                        lastNewNode.suffixLink = activeNode
                        lastNewNode = nil
                    }
                    activeLength++
                    break
                }
                
                // Split edge
                splitEnd := nextNode.start + activeLength - 1
                splitNode := &Node{
                    children: make(map[byte]*Node),
                    start:    nextNode.start,
                    end:      splitEnd,
                }
                
                activeNode.children[activeEdge] = splitNode
                nextNode.start += activeLength
                splitNode.children[st.text[nextNode.start]] = nextNode
                
                // Create new leaf node
                splitNode.children[st.text[i]] = &Node{
                    children: make(map[byte]*Node),
                    start:    i,
                    end:      n - 1,
                    isLeaf:   true,
                }
                
                if lastNewNode != nil {
                    lastNewNode.suffixLink = splitNode
                }
                
                lastNewNode = splitNode
            }
            
            remainingSuffixCount--
            
            if activeNode == st.root && activeLength > 0 {
                activeLength--
                activeEdge = st.text[i-remainingSuffixCount+1]
            } else if activeNode != st.root {
                activeNode = activeNode.suffixLink
            }
        }
    }
}

func (st *SuffixTree) Search(pattern string) []int {
    var result []int
    st.searchHelper(st.root, pattern, 0, &result)
    return result
}

func (st *SuffixTree) searchHelper(node *Node, pattern string, depth int, result *[]int) {
    if node.isLeaf {
        *result = append(*result, node.start-depth)
        return
    }
    
    if len(pattern) == 0 {
        st.collectLeaves(node, depth, result)
        return
    }
    
    child := node.children[pattern[0]]
    if child == nil {
        return
    }
    
    edgeLength := child.end - child.start + 1
    if len(pattern) <= edgeLength {
        // Pattern is within this edge
        for i := 0; i < len(pattern); i++ {
            if st.text[child.start+i] != pattern[i] {
                return
            }
        }
        st.collectLeaves(child, depth+len(pattern), result)
    } else {
        // Pattern extends beyond this edge
        for i := 0; i < edgeLength; i++ {
            if st.text[child.start+i] != pattern[i] {
                return
            }
        }
        st.searchHelper(child, pattern[edgeLength:], depth+edgeLength, result)
    }
}

func (st *SuffixTree) collectLeaves(node *Node, depth int, result *[]int) {
    if node.isLeaf {
        *result = append(*result, node.start-depth)
        return
    }
    
    for _, child := range node.children {
        st.collectLeaves(child, depth+child.end-child.start+1, result)
    }
}
```

## String Compression

### LZ77 Compression

**Problem**: Compress strings using LZ77 algorithm.

```go
// LZ77 Compression Implementation
type LZ77Compressor struct {
    windowSize int
    bufferSize int
}

type LZ77Token struct {
    Offset int
    Length int
    Next   byte
}

func NewLZ77Compressor(windowSize, bufferSize int) *LZ77Compressor {
    return &LZ77Compressor{
        windowSize: windowSize,
        bufferSize: bufferSize,
    }
}

func (lz *LZ77Compressor) Compress(data []byte) []LZ77Token {
    var tokens []LZ77Token
    n := len(data)
    
    for i := 0; i < n; i++ {
        // Find longest match in sliding window
        match := lz.findLongestMatch(data, i)
        
        if match.Length > 0 {
            tokens = append(tokens, LZ77Token{
                Offset: match.Offset,
                Length: match.Length,
                Next:   data[i+match.Length],
            })
            i += match.Length
        } else {
            tokens = append(tokens, LZ77Token{
                Offset: 0,
                Length: 0,
                Next:   data[i],
            })
        }
    }
    
    return tokens
}

func (lz *LZ77Compressor) findLongestMatch(data []byte, pos int) *Match {
    var bestMatch Match
    
    // Search in sliding window
    start := max(0, pos-lz.windowSize)
    
    for i := start; i < pos; i++ {
        length := 0
        
        // Find common prefix
        for j := 0; j < lz.bufferSize && pos+j < len(data) && i+j < pos; j++ {
            if data[i+j] == data[pos+j] {
                length++
            } else {
                break
            }
        }
        
        if length > bestMatch.Length {
            bestMatch.Offset = pos - i
            bestMatch.Length = length
        }
    }
    
    return &bestMatch
}

type Match struct {
    Offset int
    Length int
}

func (lz *LZ77Compressor) Decompress(tokens []LZ77Token) []byte {
    var result []byte
    
    for _, token := range tokens {
        if token.Length > 0 {
            // Copy from previous position
            start := len(result) - token.Offset
            for i := 0; i < token.Length; i++ {
                result = append(result, result[start+i])
            }
        }
        
        // Add next character
        result = append(result, token.Next)
    }
    
    return result
}
```

### Huffman Coding

**Problem**: Compress strings using Huffman coding.

```go
// Huffman Coding Implementation
type HuffmanNode struct {
    char   byte
    freq   int
    left   *HuffmanNode
    right  *HuffmanNode
    isLeaf bool
}

type HuffmanCoder struct {
    root     *HuffmanNode
    codes    map[byte]string
    codeMap  map[string]byte
}

func NewHuffmanCoder() *HuffmanCoder {
    return &HuffmanCoder{
        codes:   make(map[byte]string),
        codeMap: make(map[string]byte),
    }
}

func (hc *HuffmanCoder) BuildTree(data []byte) {
    // Count frequencies
    freq := make(map[byte]int)
    for _, b := range data {
        freq[b]++
    }
    
    // Create priority queue
    pq := make(PriorityQueue, 0)
    for char, f := range freq {
        pq = append(pq, &HuffmanNode{
            char:   char,
            freq:   f,
            isLeaf: true,
        })
    }
    
    heap.Init(&pq)
    
    // Build tree
    for pq.Len() > 1 {
        left := heap.Pop(&pq).(*HuffmanNode)
        right := heap.Pop(&pq).(*HuffmanNode)
        
        merged := &HuffmanNode{
            freq:   left.freq + right.freq,
            left:   left,
            right:  right,
            isLeaf: false,
        }
        
        heap.Push(&pq, merged)
    }
    
    hc.root = heap.Pop(&pq).(*HuffmanNode)
    hc.buildCodes(hc.root, "")
}

func (hc *HuffmanCoder) buildCodes(node *HuffmanNode, code string) {
    if node.isLeaf {
        hc.codes[node.char] = code
        hc.codeMap[code] = node.char
        return
    }
    
    hc.buildCodes(node.left, code+"0")
    hc.buildCodes(node.right, code+"1")
}

func (hc *HuffmanCoder) Encode(data []byte) []byte {
    var encoded strings.Builder
    
    for _, b := range data {
        encoded.WriteString(hc.codes[b])
    }
    
    // Convert binary string to bytes
    binaryStr := encoded.String()
    result := make([]byte, (len(binaryStr)+7)/8)
    
    for i := 0; i < len(binaryStr); i++ {
        if binaryStr[i] == '1' {
            result[i/8] |= 1 << (7 - i%8)
        }
    }
    
    return result
}

func (hc *HuffmanCoder) Decode(data []byte) []byte {
    var result []byte
    var current strings.Builder
    
    // Convert bytes to binary string
    var binaryStr strings.Builder
    for _, b := range data {
        for i := 7; i >= 0; i-- {
            if (b>>i)&1 == 1 {
                binaryStr.WriteString("1")
            } else {
                binaryStr.WriteString("0")
            }
        }
    }
    
    // Decode using Huffman tree
    for _, bit := range binaryStr.String() {
        current.WriteString(string(bit))
        
        if char, exists := hc.codeMap[current.String()]; exists {
            result = append(result, char)
            current.Reset()
        }
    }
    
    return result
}

// Priority Queue for Huffman nodes
type PriorityQueue []*HuffmanNode

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
    return pq[i].freq < pq[j].freq
}

func (pq PriorityQueue) Swap(i, j int) {
    pq[i], pq[j] = pq[j], pq[i]
}

func (pq *PriorityQueue) Push(x interface{}) {
    *pq = append(*pq, x.(*HuffmanNode))
}

func (pq *PriorityQueue) Pop() interface{} {
    old := *pq
    n := len(old)
    item := old[n-1]
    *pq = old[0 : n-1]
    return item
}
```

## Pattern Matching

### Regular Expression Engine

**Problem**: Implement a basic regular expression engine.

```go
// Regular Expression Engine
type RegexEngine struct {
    pattern string
    compiled *RegexNode
}

type RegexNode struct {
    type_    NodeType
    char     byte
    children []*RegexNode
    quantifier *Quantifier
}

type NodeType int

const (
    Char NodeType = iota
    Dot
    Star
    Plus
    Question
    Alternation
    Concatenation
    Group
)

type Quantifier struct {
    min int
    max int
}

func NewRegexEngine(pattern string) *RegexEngine {
    re := &RegexEngine{
        pattern: pattern,
    }
    
    re.compiled = re.compile(pattern)
    return re
}

func (re *RegexEngine) compile(pattern string) *RegexNode {
    tokens := re.tokenize(pattern)
    return re.parse(tokens)
}

func (re *RegexEngine) tokenize(pattern string) []Token {
    var tokens []Token
    
    for i := 0; i < len(pattern); i++ {
        switch pattern[i] {
        case '.':
            tokens = append(tokens, Token{Type: Dot, Value: "."})
        case '*':
            tokens = append(tokens, Token{Type: Star, Value: "*"})
        case '+':
            tokens = append(tokens, Token{Type: Plus, Value: "+"})
        case '?':
            tokens = append(tokens, Token{Type: Question, Value: "?"})
        case '|':
            tokens = append(tokens, Token{Type: Alternation, Value: "|"})
        case '(':
            tokens = append(tokens, Token{Type: LeftParen, Value: "("})
        case ')':
            tokens = append(tokens, Token{Type: RightParen, Value: ")"})
        case '\\':
            if i+1 < len(pattern) {
                i++
                tokens = append(tokens, Token{Type: Char, Value: string(pattern[i])})
            }
        default:
            tokens = append(tokens, Token{Type: Char, Value: string(pattern[i])})
        }
    }
    
    return tokens
}

type Token struct {
    Type  TokenType
    Value string
}

type TokenType int

const (
    Char TokenType = iota
    Dot
    Star
    Plus
    Question
    Alternation
    LeftParen
    RightParen
)

func (re *RegexEngine) parse(tokens []Token) *RegexNode {
    // Simplified parsing - in practice, this would be more complex
    if len(tokens) == 0 {
        return nil
    }
    
    if len(tokens) == 1 {
        return re.parseAtom(tokens[0])
    }
    
    // Handle concatenation
    var children []*RegexNode
    for _, token := range tokens {
        if token.Type != Alternation {
            children = append(children, re.parseAtom(token))
        }
    }
    
    if len(children) == 1 {
        return children[0]
    }
    
    return &RegexNode{
        type_:    Concatenation,
        children: children,
    }
}

func (re *RegexEngine) parseAtom(token Token) *RegexNode {
    switch token.Type {
    case Char:
        return &RegexNode{
            type_: Char,
            char:  token.Value[0],
        }
    case Dot:
        return &RegexNode{
            type_: Dot,
        }
    case Star:
        return &RegexNode{
            type_: Star,
        }
    case Plus:
        return &RegexNode{
            type_: Plus,
        }
    case Question:
        return &RegexNode{
            type_: Question,
        }
    default:
        return nil
    }
}

func (re *RegexEngine) Match(text string) bool {
    return re.matchHelper(re.compiled, text, 0)
}

func (re *RegexEngine) matchHelper(node *RegexNode, text string, pos int) bool {
    if node == nil {
        return pos == len(text)
    }
    
    switch node.type_ {
    case Char:
        if pos < len(text) && text[pos] == node.char {
            return re.matchHelper(node.children[0], text, pos+1)
        }
        return false
        
    case Dot:
        if pos < len(text) {
            return re.matchHelper(node.children[0], text, pos+1)
        }
        return false
        
    case Star:
        // Match zero or more
        for i := pos; i <= len(text); i++ {
            if re.matchHelper(node.children[0], text, i) {
                return true
            }
        }
        return false
        
    case Plus:
        // Match one or more
        if pos < len(text) {
            for i := pos + 1; i <= len(text); i++ {
                if re.matchHelper(node.children[0], text, i) {
                    return true
                }
            }
        }
        return false
        
    case Question:
        // Match zero or one
        return re.matchHelper(node.children[0], text, pos) ||
               re.matchHelper(node.children[0], text, pos+1)
               
    case Concatenation:
        for _, child := range node.children {
            if !re.matchHelper(child, text, pos) {
                return false
            }
            pos++
        }
        return true
        
    default:
        return false
    }
}
```

## String Processing

### String Edit Distance

**Problem**: Calculate minimum edit distance between two strings.

```go
// Edit Distance (Levenshtein Distance)
func EditDistance(s1, s2 string) int {
    m, n := len(s1), len(s2)
    
    // Create DP table
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
    
    // Fill DP table
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if s1[i-1] == s2[j-1] {
                dp[i][j] = dp[i-1][j-1]
            } else {
                dp[i][j] = 1 + min(
                    dp[i-1][j],   // deletion
                    dp[i][j-1],   // insertion
                    dp[i-1][j-1], // substitution
                )
            }
        }
    }
    
    return dp[m][n]
}

// Space-optimized version
func EditDistanceOptimized(s1, s2 string) int {
    if len(s1) < len(s2) {
        s1, s2 = s2, s1
    }
    
    prev := make([]int, len(s2)+1)
    curr := make([]int, len(s2)+1)
    
    // Initialize
    for j := 0; j <= len(s2); j++ {
        prev[j] = j
    }
    
    for i := 1; i <= len(s1); i++ {
        curr[0] = i
        
        for j := 1; j <= len(s2); j++ {
            if s1[i-1] == s2[j-1] {
                curr[j] = prev[j-1]
            } else {
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
            }
        }
        
        prev, curr = curr, prev
    }
    
    return prev[len(s2)]
}
```

### Longest Common Subsequence

**Problem**: Find longest common subsequence between two strings.

```go
// Longest Common Subsequence
func LongestCommonSubsequence(s1, s2 string) string {
    m, n := len(s1), len(s2)
    
    // Create DP table
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }
    
    // Fill DP table
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if s1[i-1] == s2[j-1] {
                dp[i][j] = 1 + dp[i-1][j-1]
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            }
        }
    }
    
    // Reconstruct LCS
    return reconstructLCS(dp, s1, s2, m, n)
}

func reconstructLCS(dp [][]int, s1, s2 string, i, j int) string {
    if i == 0 || j == 0 {
        return ""
    }
    
    if s1[i-1] == s2[j-1] {
        return reconstructLCS(dp, s1, s2, i-1, j-1) + string(s1[i-1])
    }
    
    if dp[i-1][j] > dp[i][j-1] {
        return reconstructLCS(dp, s1, s2, i-1, j)
    }
    
    return reconstructLCS(dp, s1, s2, i, j-1)
}

// Space-optimized version
func LongestCommonSubsequenceOptimized(s1, s2 string) int {
    if len(s1) < len(s2) {
        s1, s2 = s2, s1
    }
    
    prev := make([]int, len(s2)+1)
    curr := make([]int, len(s2)+1)
    
    for i := 1; i <= len(s1); i++ {
        for j := 1; j <= len(s2); j++ {
            if s1[i-1] == s2[j-1] {
                curr[j] = 1 + prev[j-1]
            } else {
                curr[j] = max(prev[j], curr[j-1])
            }
        }
        
        prev, curr = curr, prev
    }
    
    return prev[len(s2)]
}
```

## Advanced Applications

### String Matching with Wildcards

**Problem**: Match strings with wildcard characters.

```go
// Wildcard Pattern Matching
func WildcardMatch(text, pattern string) bool {
    m, n := len(text), len(pattern)
    
    // Create DP table
    dp := make([][]bool, m+1)
    for i := range dp {
        dp[i] = make([]bool, n+1)
    }
    
    // Base case: empty pattern matches empty text
    dp[0][0] = true
    
    // Handle patterns starting with '*'
    for j := 1; j <= n; j++ {
        if pattern[j-1] == '*' {
            dp[0][j] = dp[0][j-1]
        }
    }
    
    // Fill DP table
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if pattern[j-1] == '*' {
                // '*' can match zero or more characters
                dp[i][j] = dp[i][j-1] || dp[i-1][j]
            } else if pattern[j-1] == '?' || text[i-1] == pattern[j-1] {
                // '?' matches any character, or characters match
                dp[i][j] = dp[i-1][j-1]
            } else {
                dp[i][j] = false
            }
        }
    }
    
    return dp[m][n]
}

// Space-optimized version
func WildcardMatchOptimized(text, pattern string) bool {
    m, n := len(text), len(pattern)
    
    prev := make([]bool, n+1)
    curr := make([]bool, n+1)
    
    prev[0] = true
    
    for j := 1; j <= n; j++ {
        if pattern[j-1] == '*' {
            prev[j] = prev[j-1]
        }
    }
    
    for i := 1; i <= m; i++ {
        curr[0] = false
        
        for j := 1; j <= n; j++ {
            if pattern[j-1] == '*' {
                curr[j] = curr[j-1] || prev[j]
            } else if pattern[j-1] == '?' || text[i-1] == pattern[j-1] {
                curr[j] = prev[j-1]
            } else {
                curr[j] = false
            }
        }
        
        prev, curr = curr, prev
    }
    
    return prev[n]
}
```

### Palindrome Processing

**Problem**: Find longest palindromic substring.

```go
// Longest Palindromic Substring
func LongestPalindromicSubstring(s string) string {
    if len(s) == 0 {
        return ""
    }
    
    start, maxLen := 0, 1
    
    for i := 0; i < len(s); i++ {
        // Check for odd length palindromes
        len1 := expandAroundCenter(s, i, i)
        
        // Check for even length palindromes
        len2 := expandAroundCenter(s, i, i+1)
        
        maxLenCurrent := max(len1, len2)
        
        if maxLenCurrent > maxLen {
            maxLen = maxLenCurrent
            start = i - (maxLenCurrent-1)/2
        }
    }
    
    return s[start : start+maxLen]
}

func expandAroundCenter(s string, left, right int) int {
    for left >= 0 && right < len(s) && s[left] == s[right] {
        left--
        right++
    }
    
    return right - left - 1
}

// Manacher's Algorithm for O(n) solution
func LongestPalindromicSubstringManacher(s string) string {
    if len(s) == 0 {
        return ""
    }
    
    // Transform string to handle even length palindromes
    transformed := "#"
    for _, c := range s {
        transformed += string(c) + "#"
    }
    
    n := len(transformed)
    p := make([]int, n)
    center, right := 0, 0
    
    for i := 0; i < n; i++ {
        if i < right {
            p[i] = min(right-i, p[2*center-i])
        }
        
        // Try to expand palindrome centered at i
        try := p[i] + 1
        for i+try < n && i-try >= 0 && transformed[i+try] == transformed[i-try] {
            p[i]++
            try++
        }
        
        // Update center and right if we found a longer palindrome
        if i+p[i] > right {
            center = i
            right = i + p[i]
        }
    }
    
    // Find the longest palindrome
    maxLen, centerIndex := 0, 0
    for i := 0; i < n; i++ {
        if p[i] > maxLen {
            maxLen = p[i]
            centerIndex = i
        }
    }
    
    // Extract the original string
    start := (centerIndex - maxLen) / 2
    return s[start : start+maxLen]
}
```

## Conclusion

Advanced string algorithms provide:

1. **Efficiency**: Optimized algorithms for string operations
2. **Pattern Matching**: Sophisticated pattern matching techniques
3. **Compression**: String compression and encoding methods
4. **Data Structures**: Advanced data structures for string processing
5. **Applications**: Real-world applications in text processing
6. **Optimization**: Space and time complexity optimizations
7. **Scalability**: Algorithms that work with large datasets

Mastering these algorithms prepares you for complex string processing challenges in technical interviews and real-world applications.

## Additional Resources

- [String Algorithms](https://www.stringalgorithms.com/)
- [Pattern Matching](https://www.patternmatching.com/)
- [Text Processing](https://www.textprocessing.com/)
- [Compression Algorithms](https://www.compressionalgorithms.com/)
- [Suffix Structures](https://www.suffixstructures.com/)
- [Regular Expressions](https://www.regularexpressions.com/)
- [String Matching](https://www.stringmatching.com/)
- [Advanced String Processing](https://www.advancedstringprocessing.com/)
