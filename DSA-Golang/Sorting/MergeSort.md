# Merge Sort

### Problem
Implement Merge Sort algorithm to sort an array of integers.

Merge Sort is a divide-and-conquer algorithm that divides the input array into two halves, calls itself for the two halves, and then merges the two sorted halves.

**Example:**
```
Input: [64, 34, 25, 12, 22, 11, 90]
Output: [11, 12, 22, 25, 34, 64, 90]
```

### Golang Solution

```go
func mergeSort(nums []int) {
    if len(nums) <= 1 {
        return
    }
    
    mid := len(nums) / 2
    left := make([]int, mid)
    right := make([]int, len(nums)-mid)
    
    copy(left, nums[:mid])
    copy(right, nums[mid:])
    
    mergeSort(left)
    mergeSort(right)
    
    merge(nums, left, right)
}

func merge(nums, left, right []int) {
    i, j, k := 0, 0, 0
    
    for i < len(left) && j < len(right) {
        if left[i] <= right[j] {
            nums[k] = left[i]
            i++
        } else {
            nums[k] = right[j]
            j++
        }
        k++
    }
    
    for i < len(left) {
        nums[k] = left[i]
        i++
        k++
    }
    
    for j < len(right) {
        nums[k] = right[j]
        j++
        k++
    }
}
```

### Alternative Solutions

#### **In-Place Merge Sort**
```go
func mergeSortInPlace(nums []int) {
    mergeSortHelper(nums, 0, len(nums)-1)
}

func mergeSortHelper(nums []int, left, right int) {
    if left < right {
        mid := left + (right-left)/2
        
        mergeSortHelper(nums, left, mid)
        mergeSortHelper(nums, mid+1, right)
        
        mergeInPlace(nums, left, mid, right)
    }
}

func mergeInPlace(nums []int, left, mid, right int) {
    leftSize := mid - left + 1
    rightSize := right - mid
    
    leftArr := make([]int, leftSize)
    rightArr := make([]int, rightSize)
    
    copy(leftArr, nums[left:left+leftSize])
    copy(rightArr, nums[mid+1:mid+1+rightSize])
    
    i, j, k := 0, 0, left
    
    for i < leftSize && j < rightSize {
        if leftArr[i] <= rightArr[j] {
            nums[k] = leftArr[i]
            i++
        } else {
            nums[k] = rightArr[j]
            j++
        }
        k++
    }
    
    for i < leftSize {
        nums[k] = leftArr[i]
        i++
        k++
    }
    
    for j < rightSize {
        nums[k] = rightArr[j]
        j++
        k++
    }
}
```

#### **Iterative Merge Sort**
```go
func mergeSortIterative(nums []int) {
    n := len(nums)
    
    for size := 1; size < n; size *= 2 {
        for left := 0; left < n-1; left += 2 * size {
            mid := min(left+size-1, n-1)
            right := min(left+2*size-1, n-1)
            
            merge(nums, left, mid, right)
        }
    }
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

#### **Merge Sort with Counters**
```go
func mergeSortWithCounters(nums []int) ([]int, int, int) {
    comparisons := 0
    swaps := 0
    
    mergeSortHelperWithCounters(nums, 0, len(nums)-1, &comparisons, &swaps)
    
    return nums, comparisons, swaps
}

func mergeSortHelperWithCounters(nums []int, left, right int, comparisons, swaps *int) {
    if left < right {
        mid := left + (right-left)/2
        
        mergeSortHelperWithCounters(nums, left, mid, comparisons, swaps)
        mergeSortHelperWithCounters(nums, mid+1, right, comparisons, swaps)
        
        mergeWithCounters(nums, left, mid, right, comparisons, swaps)
    }
}

func mergeWithCounters(nums []int, left, mid, right int, comparisons, swaps *int) {
    leftSize := mid - left + 1
    rightSize := right - mid
    
    leftArr := make([]int, leftSize)
    rightArr := make([]int, rightSize)
    
    copy(leftArr, nums[left:left+leftSize])
    copy(rightArr, nums[mid+1:mid+1+rightSize])
    
    i, j, k := 0, 0, left
    
    for i < leftSize && j < rightSize {
        (*comparisons)++
        if leftArr[i] <= rightArr[j] {
            nums[k] = leftArr[i]
            i++
        } else {
            nums[k] = rightArr[j]
            j++
        }
        (*swaps)++
        k++
    }
    
    for i < leftSize {
        nums[k] = leftArr[i]
        i++
        k++
    }
    
    for j < rightSize {
        nums[k] = rightArr[j]
        j++
        k++
    }
}
```

#### **Natural Merge Sort**
```go
func naturalMergeSort(nums []int) {
    n := len(nums)
    
    for {
        runs := findRuns(nums)
        if len(runs) == 1 {
            break
        }
        
        for i := 0; i < len(runs)-1; i += 2 {
            left := runs[i]
            right := runs[i+1]
            merge(nums, left.start, left.end, right.end)
        }
    }
}

type Run struct {
    start, end int
}

func findRuns(nums []int) []Run {
    var runs []Run
    start := 0
    
    for i := 1; i < len(nums); i++ {
        if nums[i] < nums[i-1] {
            runs = append(runs, Run{start, i - 1})
            start = i
        }
    }
    
    runs = append(runs, Run{start, len(nums) - 1})
    return runs
}
```

### Complexity
- **Time Complexity:** O(n log n) in all cases
- **Space Complexity:** O(n)