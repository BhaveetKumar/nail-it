# Search a 2D Matrix

### Problem
Write an efficient algorithm that searches for a value `target` in an `m x n` integer matrix. This matrix has the following properties:

- Integers in each row are sorted from left to right.
- The first integer of each row is greater than the last integer of the previous row.

**Example:**
```
Input: matrix = [[1,4,7,11],[2,5,8,12],[3,6,9,16],[10,13,14,17]], target = 5
Output: true

Input: matrix = [[1,4,7,11],[2,5,8,12],[3,6,9,16],[10,13,14,17]], target = 3
Output: false
```

### Golang Solution

```go
func searchMatrix(matrix [][]int, target int) bool {
    if len(matrix) == 0 || len(matrix[0]) == 0 {
        return false
    }
    
    m, n := len(matrix), len(matrix[0])
    left, right := 0, m*n-1
    
    for left <= right {
        mid := left + (right-left)/2
        midValue := matrix[mid/n][mid%n]
        
        if midValue == target {
            return true
        } else if midValue < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return false
}
```

### Alternative Solutions

#### **Two-Step Binary Search**
```go
func searchMatrixTwoStep(matrix [][]int, target int) bool {
    if len(matrix) == 0 || len(matrix[0]) == 0 {
        return false
    }
    
    // First, find the correct row
    row := findRow(matrix, target)
    if row == -1 {
        return false
    }
    
    // Then, search in that row
    return binarySearch(matrix[row], target)
}

func findRow(matrix [][]int, target int) int {
    left, right := 0, len(matrix)-1
    
    for left <= right {
        mid := left + (right-left)/2
        
        if matrix[mid][0] <= target && target <= matrix[mid][len(matrix[mid])-1] {
            return mid
        } else if matrix[mid][0] > target {
            right = mid - 1
        } else {
            left = mid + 1
        }
    }
    
    return -1
}

func binarySearch(nums []int, target int) bool {
    left, right := 0, len(nums)-1
    
    for left <= right {
        mid := left + (right-left)/2
        
        if nums[mid] == target {
            return true
        } else if nums[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return false
}
```

#### **Linear Search (Not Recommended)**
```go
func searchMatrixLinear(matrix [][]int, target int) bool {
    for i := 0; i < len(matrix); i++ {
        for j := 0; j < len(matrix[i]); j++ {
            if matrix[i][j] == target {
                return true
            }
        }
    }
    return false
}
```

### Complexity
- **Time Complexity:** O(log(m × n)) for single binary search, O(log m + log n) for two-step
- **Space Complexity:** O(1)
