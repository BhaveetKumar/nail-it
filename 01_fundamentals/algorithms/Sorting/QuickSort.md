---
# Auto-generated front matter
Title: Quicksort
LastUpdated: 2025-11-06T20:45:58.735739
Tags: []
Status: draft
---

# Quick Sort

### Problem
Implement Quick Sort algorithm to sort an array of integers.

Quick Sort is a divide-and-conquer algorithm that picks an element as pivot and partitions the given array around the picked pivot.

**Example:**
```
Input: [64, 34, 25, 12, 22, 11, 90]
Output: [11, 12, 22, 25, 34, 64, 90]
```

### Golang Solution

```go
func quickSort(nums []int) {
    if len(nums) <= 1 {
        return
    }
    
    quickSortHelper(nums, 0, len(nums)-1)
}

func quickSortHelper(nums []int, low, high int) {
    if low < high {
        pivotIndex := partition(nums, low, high)
        quickSortHelper(nums, low, pivotIndex-1)
        quickSortHelper(nums, pivotIndex+1, high)
    }
}

func partition(nums []int, low, high int) int {
    pivot := nums[high]
    i := low - 1
    
    for j := low; j < high; j++ {
        if nums[j] <= pivot {
            i++
            nums[i], nums[j] = nums[j], nums[i]
        }
    }
    
    nums[i+1], nums[high] = nums[high], nums[i+1]
    return i + 1
}
```

### Alternative Solutions

#### **Using First Element as Pivot**
```go
func quickSortFirstPivot(nums []int) {
    if len(nums) <= 1 {
        return
    }
    
    quickSortFirstPivotHelper(nums, 0, len(nums)-1)
}

func quickSortFirstPivotHelper(nums []int, low, high int) {
    if low < high {
        pivotIndex := partitionFirstPivot(nums, low, high)
        quickSortFirstPivotHelper(nums, low, pivotIndex-1)
        quickSortFirstPivotHelper(nums, pivotIndex+1, high)
    }
}

func partitionFirstPivot(nums []int, low, high int) int {
    pivot := nums[low]
    i := low + 1
    j := high
    
    for i <= j {
        for i <= j && nums[i] <= pivot {
            i++
        }
        for i <= j && nums[j] > pivot {
            j--
        }
        if i < j {
            nums[i], nums[j] = nums[j], nums[i]
        }
    }
    
    nums[low], nums[j] = nums[j], nums[low]
    return j
}
```

#### **Using Middle Element as Pivot**
```go
func quickSortMiddlePivot(nums []int) {
    if len(nums) <= 1 {
        return
    }
    
    quickSortMiddlePivotHelper(nums, 0, len(nums)-1)
}

func quickSortMiddlePivotHelper(nums []int, low, high int) {
    if low < high {
        pivotIndex := partitionMiddlePivot(nums, low, high)
        quickSortMiddlePivotHelper(nums, low, pivotIndex-1)
        quickSortMiddlePivotHelper(nums, pivotIndex+1, high)
    }
}

func partitionMiddlePivot(nums []int, low, high int) int {
    mid := low + (high-low)/2
    nums[mid], nums[high] = nums[high], nums[mid]
    
    pivot := nums[high]
    i := low - 1
    
    for j := low; j < high; j++ {
        if nums[j] <= pivot {
            i++
            nums[i], nums[j] = nums[j], nums[i]
        }
    }
    
    nums[i+1], nums[high] = nums[high], nums[i+1]
    return i + 1
}
```

#### **Iterative Quick Sort**
```go
func quickSortIterative(nums []int) {
    if len(nums) <= 1 {
        return
    }
    
    stack := [][]int{{0, len(nums) - 1}}
    
    for len(stack) > 0 {
        low, high := stack[len(stack)-1][0], stack[len(stack)-1][1]
        stack = stack[:len(stack)-1]
        
        if low < high {
            pivotIndex := partition(nums, low, high)
            stack = append(stack, []int{low, pivotIndex - 1})
            stack = append(stack, []int{pivotIndex + 1, high})
        }
    }
}

func partition(nums []int, low, high int) int {
    pivot := nums[high]
    i := low - 1
    
    for j := low; j < high; j++ {
        if nums[j] <= pivot {
            i++
            nums[i], nums[j] = nums[j], nums[i]
        }
    }
    
    nums[i+1], nums[high] = nums[high], nums[i+1]
    return i + 1
}
```

#### **Quick Sort with Counters**
```go
func quickSortWithCounters(nums []int) ([]int, int, int) {
    comparisons := 0
    swaps := 0
    
    quickSortWithCountersHelper(nums, 0, len(nums)-1, &comparisons, &swaps)
    
    return nums, comparisons, swaps
}

func quickSortWithCountersHelper(nums []int, low, high int, comparisons, swaps *int) {
    if low < high {
        pivotIndex := partitionWithCounters(nums, low, high, comparisons, swaps)
        quickSortWithCountersHelper(nums, low, pivotIndex-1, comparisons, swaps)
        quickSortWithCountersHelper(nums, pivotIndex+1, high, comparisons, swaps)
    }
}

func partitionWithCounters(nums []int, low, high int, comparisons, swaps *int) int {
    pivot := nums[high]
    i := low - 1
    
    for j := low; j < high; j++ {
        (*comparisons)++
        if nums[j] <= pivot {
            i++
            nums[i], nums[j] = nums[j], nums[i]
            (*swaps)++
        }
    }
    
    nums[i+1], nums[high] = nums[high], nums[i+1]
    (*swaps)++
    return i + 1
}
```

#### **Three-Way Quick Sort**
```go
func threeWayQuickSort(nums []int) {
    if len(nums) <= 1 {
        return
    }
    
    threeWayQuickSortHelper(nums, 0, len(nums)-1)
}

func threeWayQuickSortHelper(nums []int, low, high int) {
    if low >= high {
        return
    }
    
    lt, gt := threeWayPartition(nums, low, high)
    threeWayQuickSortHelper(nums, low, lt-1)
    threeWayQuickSortHelper(nums, gt+1, high)
}

func threeWayPartition(nums []int, low, high int) (int, int) {
    pivot := nums[low]
    lt := low
    gt := high
    i := low + 1
    
    for i <= gt {
        if nums[i] < pivot {
            nums[lt], nums[i] = nums[i], nums[lt]
            lt++
            i++
        } else if nums[i] > pivot {
            nums[i], nums[gt] = nums[gt], nums[i]
            gt--
        } else {
            i++
        }
    }
    
    return lt, gt
}
```

### Complexity
- **Time Complexity:** O(n log n) average case, O(n²) worst case
- **Space Complexity:** O(log n) average case, O(n) worst case