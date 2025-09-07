# Selection Sort

### Problem
Implement Selection Sort algorithm to sort an array of integers.

Selection Sort is a simple sorting algorithm that repeatedly finds the minimum element from the unsorted portion and moves it to the beginning.

**Example:**
```
Input: [64, 25, 12, 22, 11]
Output: [11, 12, 22, 25, 64]
```

### Golang Solution

```go
func selectionSort(nums []int) {
    n := len(nums)
    
    for i := 0; i < n-1; i++ {
        minIndex := i
        
        // Find minimum element in remaining array
        for j := i + 1; j < n; j++ {
            if nums[j] < nums[minIndex] {
                minIndex = j
            }
        }
        
        // Swap if minimum is not at current position
        if minIndex != i {
            nums[i], nums[minIndex] = nums[minIndex], nums[i]
        }
    }
}
```

### Alternative Solutions

#### **Recursive Approach**
```go
func selectionSortRecursive(nums []int) {
    selectionSortHelper(nums, 0)
}

func selectionSortHelper(nums []int, start int) {
    if start >= len(nums)-1 {
        return
    }
    
    minIndex := start
    
    // Find minimum element
    for i := start + 1; i < len(nums); i++ {
        if nums[i] < nums[minIndex] {
            minIndex = i
        }
    }
    
    // Swap if necessary
    if minIndex != start {
        nums[start], nums[minIndex] = nums[minIndex], nums[start]
    }
    
    // Recursively sort remaining array
    selectionSortHelper(nums, start+1)
}
```

#### **Stable Selection Sort**
```go
func stableSelectionSort(nums []int) {
    n := len(nums)
    
    for i := 0; i < n-1; i++ {
        minIndex := i
        
        // Find minimum element
        for j := i + 1; j < n; j++ {
            if nums[j] < nums[minIndex] {
                minIndex = j
            }
        }
        
        // Move minimum element to correct position
        // by shifting elements to the right
        minValue := nums[minIndex]
        for k := minIndex; k > i; k-- {
            nums[k] = nums[k-1]
        }
        nums[i] = minValue
    }
}
```

#### **Selection Sort with Counters**
```go
func selectionSortWithCounters(nums []int) ([]int, int, int) {
    n := len(nums)
    comparisons := 0
    swaps := 0
    
    for i := 0; i < n-1; i++ {
        minIndex := i
        
        for j := i + 1; j < n; j++ {
            comparisons++
            if nums[j] < nums[minIndex] {
                minIndex = j
            }
        }
        
        if minIndex != i {
            nums[i], nums[minIndex] = nums[minIndex], nums[i]
            swaps++
        }
    }
    
    return nums, comparisons, swaps
}
```

#### **Generic Selection Sort**
```go
func selectionSortGeneric[T comparable](nums []T, less func(T, T) bool) {
    n := len(nums)
    
    for i := 0; i < n-1; i++ {
        minIndex := i
        
        for j := i + 1; j < n; j++ {
            if less(nums[j], nums[minIndex]) {
                minIndex = j
            }
        }
        
        if minIndex != i {
            nums[i], nums[minIndex] = nums[minIndex], nums[i]
        }
    }
}
```

#### **Selection Sort for Strings**
```go
func selectionSortStrings(strs []string) {
    n := len(strs)
    
    for i := 0; i < n-1; i++ {
        minIndex := i
        
        for j := i + 1; j < n; j++ {
            if strs[j] < strs[minIndex] {
                minIndex = j
            }
        }
        
        if minIndex != i {
            strs[i], strs[minIndex] = strs[minIndex], strs[i]
        }
    }
}
```

### Complexity
- **Time Complexity:** O(n²) in all cases
- **Space Complexity:** O(1)
