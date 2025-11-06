---
# Auto-generated front matter
Title: Maximumxoroftwonumbersinarray
LastUpdated: 2025-11-06T20:45:58.690843
Tags: []
Status: draft
---

# Maximum XOR of Two Numbers in an Array

### Problem
Given an integer array `nums`, return the maximum result of `nums[i] XOR nums[j]`, where `0 <= i <= j < n`.

**Example:**
```
Input: nums = [3,10,5,25,2,8]
Output: 28
Explanation: The maximum result is 5 XOR 25 = 28.

Input: nums = [14,70,53,83,49,91,36,80,92,51,66,70]
Output: 127
```

### Golang Solution

```go
type TrieNode struct {
    children [2]*TrieNode
}

func findMaximumXOR(nums []int) int {
    if len(nums) < 2 {
        return 0
    }
    
    root := &TrieNode{}
    
    // Build trie
    for _, num := range nums {
        insert(root, num)
    }
    
    maxXOR := 0
    
    // Find maximum XOR for each number
    for _, num := range nums {
        maxXOR = max(maxXOR, findMaxXOR(root, num))
    }
    
    return maxXOR
}

func insert(root *TrieNode, num int) {
    node := root
    for i := 31; i >= 0; i-- {
        bit := (num >> i) & 1
        if node.children[bit] == nil {
            node.children[bit] = &TrieNode{}
        }
        node = node.children[bit]
    }
}

func findMaxXOR(root *TrieNode, num int) int {
    node := root
    maxXOR := 0
    
    for i := 31; i >= 0; i-- {
        bit := (num >> i) & 1
        oppositeBit := 1 - bit
        
        if node.children[oppositeBit] != nil {
            maxXOR |= (1 << i)
            node = node.children[oppositeBit]
        } else {
            node = node.children[bit]
        }
    }
    
    return maxXOR
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

### Alternative Solutions

#### **Brute Force**
```go
func findMaximumXORBruteForce(nums []int) int {
    maxXOR := 0
    
    for i := 0; i < len(nums); i++ {
        for j := i; j < len(nums); j++ {
            maxXOR = max(maxXOR, nums[i]^nums[j])
        }
    }
    
    return maxXOR
}
```

#### **Using Hash Set**
```go
func findMaximumXORHashSet(nums []int) int {
    maxXOR := 0
    mask := 0
    
    for i := 31; i >= 0; i-- {
        mask |= (1 << i)
        prefixes := make(map[int]bool)
        
        for _, num := range nums {
            prefixes[num&mask] = true
        }
        
        temp := maxXOR | (1 << i)
        
        for prefix := range prefixes {
            if prefixes[prefix^temp] {
                maxXOR = temp
                break
            }
        }
    }
    
    return maxXOR
}
```

#### **Bit Manipulation with Set**
```go
func findMaximumXORBitManipulation(nums []int) int {
    maxXOR := 0
    mask := 0
    
    for i := 31; i >= 0; i-- {
        mask |= (1 << i)
        set := make(map[int]bool)
        
        for _, num := range nums {
            set[num&mask] = true
        }
        
        candidate := maxXOR | (1 << i)
        
        for prefix := range set {
            if set[prefix^candidate] {
                maxXOR = candidate
                break
            }
        }
    }
    
    return maxXOR
}
```

### Complexity
- **Time Complexity:** O(n × 32) for trie, O(n²) for brute force
- **Space Complexity:** O(n × 32) for trie, O(1) for brute force
