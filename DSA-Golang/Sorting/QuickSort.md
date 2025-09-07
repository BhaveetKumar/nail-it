# Quick Sort

### Problem
Implement Quick Sort algorithm to sort an array of integers.

Quick Sort is a divide-and-conquer algorithm that works by selecting a 'pivot' element from the array and partitioning the other elements into two sub-arrays, according to whether they are less than or greater than the pivot.

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
    
    pivot := partition(nums)
    quickSort(nums[:pivot])
    quickSort(nums[pivot+1:])
}

func partition(nums []int) int {
    // Choose last element as pivot
    pivot := nums[len(nums)-1]
    i := 0 // Index of smaller element
    
    for j := 0; j < len(nums)-1; j++ {
        if nums[j] <= pivot {
            nums[i], nums[j] = nums[j], nums[i]
            i++
        }
    }
    
    // Place pivot in correct position
    nums[i], nums[len(nums)-1] = nums[len(nums)-1], nums[i]
    return i
}
```

### Alternative Solutions

#### **Randomized Quick Sort**
```go
import "math/rand"

func quickSortRandomized(nums []int) {
    if len(nums) <= 1 {
        return
    }
    
    pivot := partitionRandomized(nums)
    quickSortRandomized(nums[:pivot])
    quickSortRandomized(nums[pivot+1:])
}

func partitionRandomized(nums []int) int {
    // Choose random pivot
    randomIndex := rand.Intn(len(nums))
    nums[randomIndex], nums[len(nums)-1] = nums[len(nums)-1], nums[randomIndex]
    
    return partition(nums)
}
```

### Complexity
- **Time Complexity:** O(n log n) average, O(n²) worst case
- **Space Complexity:** O(log n) average, O(n) worst case
