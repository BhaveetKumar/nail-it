---
# Auto-generated front matter
Title: Implementtrie
LastUpdated: 2025-11-06T20:45:58.686331
Tags: []
Status: draft
---

# Implement Trie (Prefix Tree)

### Problem
A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and search strings in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:
- `Trie()` Initializes the trie object.
- `void insert(String word)` Inserts the string word into the trie.
- `boolean search(String word)` Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
- `boolean startsWith(String prefix)` Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.

**Example:**
```
Input
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]

Output
[null, null, true, false, true, null, true]
```

### Golang Solution

```go
type TrieNode struct {
    children map[rune]*TrieNode
    isEnd    bool
}

type Trie struct {
    root *TrieNode
}

func Constructor() Trie {
    return Trie{
        root: &TrieNode{
            children: make(map[rune]*TrieNode),
            isEnd:    false,
        },
    }
}

func (t *Trie) Insert(word string) {
    node := t.root
    for _, char := range word {
        if _, exists := node.children[char]; !exists {
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
        if _, exists := node.children[char]; !exists {
            return false
        }
        node = node.children[char]
    }
    return node.isEnd
}

func (t *Trie) StartsWith(prefix string) bool {
    node := t.root
    for _, char := range prefix {
        if _, exists := node.children[char]; !exists {
            return false
        }
        node = node.children[char]
    }
    return true
}
```

### Complexity
- **Time Complexity:** O(m) for insert/search/startsWith where m is word length
- **Space Complexity:** O(ALPHABET_SIZE * N * M) where N is number of words, M is average length
