---
# Auto-generated front matter
Title: Insertionsort
LastUpdated: 2025-11-06T20:45:58.734406
Tags: []
Status: draft
---

# Insertion Sort

### Problem
Implement Insertion Sort algorithm to sort an array of integers.

Insertion Sort is a simple sorting algorithm that builds the final sorted array one item at a time. It is much less efficient on large lists than more advanced algorithms such as quicksort, heapsort, or merge sort.

**Example:**
```
Input: [64, 34, 25, 12, 22, 11, 90]
Output: [11, 12, 22, 25, 34, 64, 90]
```

### Golang Solution

```go
func insertionSort(nums []int) {
    for i := 1; i < len(nums); i++ {
        key := nums[i]
        j := i - 1
        
        // Move elements greater than key one position ahead
        for j >= 0 && nums[j] > key {
            nums[j+1] = nums[j]
            j--
        }
        
        nums[j+1] = key
    }
}
```

### Alternative Solutions

#### **Recursive Approach**
```go
func insertionSortRecursive(nums []int) {
    if len(nums) <= 1 {
        return
    }
    
    // Sort first n-1 elements
    insertionSortRecursive(nums[:len(nums)-1])
    
    // Insert last element at correct position
    insertLastElement(nums)
}

func insertLastElement(nums []int) {
    key := nums[len(nums)-1]
    j := len(nums) - 2
    
    for j >= 0 && nums[j] > key {
        nums[j+1] = nums[j]
        j--
    }
    
    nums[j+1] = key
}
```

#### **Binary Search Optimization**
```go
func insertionSortBinarySearch(nums []int) {
    for i := 1; i < len(nums); i++ {
        key := nums[i]
        
        // Find position to insert using binary search
        pos := binarySearchPosition(nums, 0, i-1, key)
        
        // Shift elements to make space
        for j := i; j > pos; j-- {
            nums[j] = nums[j-1]
        }
        
        nums[pos] = key
    }
}

func binarySearchPosition(nums []int, left, right, key int) int {
    for left <= right {
        mid := left + (right-left)/2
        
        if nums[mid] <= key {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return left
}
```

#### **For Linked Lists**
```go
type ListNode struct {
    Val  int
    Next *ListNode
}

func insertionSortList(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
    
    dummy := &ListNode{Next: head}
    current := head.Next
    
    for current != nil {
        if current.Val >= head.Val {
            head = current
            current = current.Next
        } else {
            head.Next = current.Next
            
            // Find insertion position
            prev := dummy
            for prev.Next != nil && prev.Next.Val < current.Val {
                prev = prev.Next
            }
            
            // Insert current node
            current.Next = prev.Next
            prev.Next = current
            
            current = head.Next
        }
    }
    
    return dummy.Next
}
```

#### **Stable Sort Implementation**
```go
func insertionSortStable(nums []int) {
    for i := 1; i < len(nums); i++ {
        key := nums[i]
        j := i - 1
        
        // Move elements greater than key one position ahead
        // Use >= for stable sort (maintains relative order of equal elements)
        for j >= 0 && nums[j] > key {
            nums[j+1] = nums[j]
            j--
        }
        
        nums[j+1] = key
    }
}
```

#### **With Comparison Counter**
```go
func insertionSortWithCount(nums []int) ([]int, int) {
    comparisons := 0
    
    for i := 1; i < len(nums); i++ {
        key := nums[i]
        j := i - 1
        
        for j >= 0 {
            comparisons++
            if nums[j] <= key {
                break
            }
            nums[j+1] = nums[j]
            j--
        }
        
        nums[j+1] = key
    }
    
    return nums, comparisons
}
```

### Complexity
- **Time Complexity:** O(n²) worst case, O(n) best case
- **Space Complexity:** O(1)
