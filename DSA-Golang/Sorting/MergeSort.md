# Merge Sort

### Problem
Implement Merge Sort algorithm to sort an array of integers.

Merge Sort is a divide-and-conquer algorithm that works by dividing the array into two halves, sorting them separately, and then merging them back together.

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

#### **In-place Merge Sort**
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
    
    for i := 0; i < leftSize; i++ {
        leftArr[i] = nums[left+i]
    }
    for j := 0; j < rightSize; j++ {
        rightArr[j] = nums[mid+1+j]
    }
    
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

### Complexity
- **Time Complexity:** O(n log n)
- **Space Complexity:** O(n)
