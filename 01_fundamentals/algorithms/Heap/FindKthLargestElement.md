---
# Auto-generated front matter
Title: Findkthlargestelement
LastUpdated: 2025-11-06T20:45:58.718558
Tags: []
Status: draft
---

# Find Kth Largest Element in an Array

### Problem
Given an integer array `nums` and an integer `k`, return the `kth` largest element in the array.

Note that it is the `kth` largest element in sorted order, not the `kth` distinct element.

**Example:**
```
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5

Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4
```

### Golang Solution

```go
import "container/heap"

type IntHeap []int

func (h IntHeap) Len() int           { return len(h) }
func (h IntHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h IntHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *IntHeap) Push(x interface{}) {
    *h = append(*h, x.(int))
}

func (h *IntHeap) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[0 : n-1]
    return x
}

func findKthLargest(nums []int, k int) int {
    h := &IntHeap{}
    heap.Init(h)
    
    for _, num := range nums {
        heap.Push(h, num)
        if h.Len() > k {
            heap.Pop(h)
        }
    }
    
    return (*h)[0]
}
```

### Alternative Solutions

#### **Using Quick Select**
```go
func findKthLargestQuickSelect(nums []int, k int) int {
    return quickSelect(nums, 0, len(nums)-1, len(nums)-k)
}

func quickSelect(nums []int, left, right, k int) int {
    if left == right {
        return nums[left]
    }
    
    pivotIndex := partition(nums, left, right)
    
    if k == pivotIndex {
        return nums[k]
    } else if k < pivotIndex {
        return quickSelect(nums, left, pivotIndex-1, k)
    } else {
        return quickSelect(nums, pivotIndex+1, right, k)
    }
}

func partition(nums []int, left, right int) int {
    pivot := nums[right]
    i := left
    
    for j := left; j < right; j++ {
        if nums[j] <= pivot {
            nums[i], nums[j] = nums[j], nums[i]
            i++
        }
    }
    
    nums[i], nums[right] = nums[right], nums[i]
    return i
}
```

#### **Using Sorting**
```go
import "sort"

func findKthLargestSort(nums []int, k int) int {
    sort.Ints(nums)
    return nums[len(nums)-k]
}
```

#### **Return with Statistics**
```go
type KthLargestResult struct {
    KthLargest    int
    AllLargest    []int
    Position      int
    Count         int
    MinValue      int
    MaxValue      int
    AvgValue      float64
}

func findKthLargestWithStats(nums []int, k int) KthLargestResult {
    if k <= 0 || k > len(nums) {
        return KthLargestResult{}
    }
    
    // Create a copy to avoid modifying original
    sorted := make([]int, len(nums))
    copy(sorted, nums)
    sort.Ints(sorted)
    
    kthLargest := sorted[len(sorted)-k]
    
    // Find all occurrences of kth largest
    var allLargest []int
    count := 0
    position := -1
    
    for i := len(sorted) - 1; i >= 0; i-- {
        if sorted[i] == kthLargest {
            allLargest = append(allLargest, sorted[i])
            count++
            if position == -1 {
                position = i
            }
        }
    }
    
    minValue := sorted[0]
    maxValue := sorted[len(sorted)-1]
    sum := 0
    for _, num := range sorted {
        sum += num
    }
    
    return KthLargestResult{
        KthLargest: kthLargest,
        AllLargest: allLargest,
        Position:   position,
        Count:      count,
        MinValue:   minValue,
        MaxValue:   maxValue,
        AvgValue:   float64(sum) / float64(len(sorted)),
    }
}
```

#### **Return All K Largest Elements**
```go
func findKLargest(nums []int, k int) []int {
    if k <= 0 || k > len(nums) {
        return []int{}
    }
    
    // Create a copy to avoid modifying original
    sorted := make([]int, len(nums))
    copy(sorted, nums)
    sort.Ints(sorted)
    
    var result []int
    for i := len(sorted) - k; i < len(sorted); i++ {
        result = append(result, sorted[i])
    }
    
    return result
}
```

#### **Return with Frequency**
```go
type ElementFrequency struct {
    Value     int
    Frequency int
    Positions []int
}

type KthLargestWithFreq struct {
    KthLargest int
    Frequency  int
    AllFreqs   []ElementFrequency
}

func findKthLargestWithFrequency(nums []int, k int) KthLargestWithFreq {
    if k <= 0 || k > len(nums) {
        return KthLargestWithFreq{}
    }
    
    // Create a copy to avoid modifying original
    sorted := make([]int, len(nums))
    copy(sorted, nums)
    sort.Ints(sorted)
    
    kthLargest := sorted[len(sorted)-k]
    
    // Count frequencies
    freqMap := make(map[int]int)
    posMap := make(map[int][]int)
    
    for i, num := range nums {
        freqMap[num]++
        posMap[num] = append(posMap[num], i)
    }
    
    var allFreqs []ElementFrequency
    for value, freq := range freqMap {
        allFreqs = append(allFreqs, ElementFrequency{
            Value:     value,
            Frequency: freq,
            Positions: posMap[value],
        })
    }
    
    // Sort by frequency
    sort.Slice(allFreqs, func(i, j int) bool {
        return allFreqs[i].Frequency > allFreqs[j].Frequency
    })
    
    return KthLargestWithFreq{
        KthLargest: kthLargest,
        Frequency:  freqMap[kthLargest],
        AllFreqs:   allFreqs,
    }
}
```

#### **Return with Ranking**
```go
type ElementRank struct {
    Value int
    Rank  int
    Count int
}

type KthLargestWithRanking struct {
    KthLargest int
    Rank       int
    AllRanks   []ElementRank
}

func findKthLargestWithRanking(nums []int, k int) KthLargestWithRanking {
    if k <= 0 || k > len(nums) {
        return KthLargestWithRanking{}
    }
    
    // Create a copy to avoid modifying original
    sorted := make([]int, len(nums))
    copy(sorted, nums)
    sort.Ints(sorted)
    
    kthLargest := sorted[len(sorted)-k]
    
    // Create ranking
    var allRanks []ElementRank
    rank := 1
    count := 1
    
    for i := len(sorted) - 1; i >= 0; i-- {
        if i < len(sorted)-1 && sorted[i] != sorted[i+1] {
            rank += count
            count = 1
        } else if i < len(sorted)-1 {
            count++
        }
        
        allRanks = append(allRanks, ElementRank{
            Value: sorted[i],
            Rank:  rank,
            Count: count,
        })
    }
    
    // Find rank of kth largest
    kthRank := 1
    for _, rank := range allRanks {
        if rank.Value == kthLargest {
            kthRank = rank.Rank
            break
        }
    }
    
    return KthLargestWithRanking{
        KthLargest: kthLargest,
        Rank:       kthRank,
        AllRanks:   allRanks,
    }
}
```

### Complexity
- **Time Complexity:** O(n log k) for heap, O(n) average for quick select, O(n log n) for sorting
- **Space Complexity:** O(k) for heap, O(1) for quick select, O(n) for sorting