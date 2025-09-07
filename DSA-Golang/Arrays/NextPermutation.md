# Next Permutation

### Problem
A permutation of an array of integers is an arrangement of its members into a sequence or linear order.

For example, for `arr = [1,2,3]`, the following are all the permutations of `arr`: `[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]`.

The next permutation of an array of integers is the next lexicographically greater permutation of its integer.

Given an array of integers `nums`, find the next permutation of `nums`.

The replacement must be in place and use only constant extra memory.

**Example:**
```
Input: nums = [1,2,3]
Output: [1,3,2]

Input: nums = [3,2,1]
Output: [1,2,3]

Input: nums = [1,1,5]
Output: [1,5,1]
```

### Golang Solution

```go
func nextPermutation(nums []int) {
    n := len(nums)
    i := n - 2
    
    // Step 1: Find the largest index i such that nums[i] < nums[i+1]
    for i >= 0 && nums[i] >= nums[i+1] {
        i--
    }
    
    if i >= 0 {
        // Step 2: Find the largest index j such that nums[i] < nums[j]
        j := n - 1
        for nums[j] <= nums[i] {
            j--
        }
        
        // Step 3: Swap nums[i] and nums[j]
        nums[i], nums[j] = nums[j], nums[i]
    }
    
    // Step 4: Reverse the suffix starting at nums[i+1]
    reverse(nums, i+1, n-1)
}

func reverse(nums []int, start, end int) {
    for start < end {
        nums[start], nums[end] = nums[end], nums[start]
        start++
        end--
    }
}
```

### Alternative Solutions

#### **Using Library Sort**
```go
import "sort"

func nextPermutationSort(nums []int) {
    n := len(nums)
    i := n - 2
    
    // Find the pivot
    for i >= 0 && nums[i] >= nums[i+1] {
        i--
    }
    
    if i >= 0 {
        // Find the successor
        j := n - 1
        for nums[j] <= nums[i] {
            j--
        }
        
        // Swap
        nums[i], nums[j] = nums[j], nums[i]
    }
    
    // Reverse suffix
    sort.Ints(nums[i+1:])
}
```

#### **Generate All Permutations (Not Recommended)**
```go
func nextPermutationAll(nums []int) {
    // This approach generates all permutations and finds the next one
    // Not efficient for large arrays
    permutations := generatePermutations(nums)
    currentIndex := findPermutationIndex(permutations, nums)
    
    if currentIndex < len(permutations)-1 {
        copy(nums, permutations[currentIndex+1])
    } else {
        copy(nums, permutations[0]) // Wrap around to first
    }
}

func generatePermutations(nums []int) [][]int {
    // Implementation would generate all permutations
    // This is O(n!) and not practical for large inputs
    return [][]int{}
}

func findPermutationIndex(permutations [][]int, target []int) int {
    for i, perm := range permutations {
        if equal(perm, target) {
            return i
        }
    }
    return -1
}

func equal(a, b []int) bool {
    if len(a) != len(b) {
        return false
    }
    for i := range a {
        if a[i] != b[i] {
            return false
        }
    }
    return true
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)