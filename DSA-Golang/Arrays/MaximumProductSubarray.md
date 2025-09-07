# Maximum Product Subarray

### Problem
Given an integer array `nums`, find a contiguous non-empty subarray within the array that has the largest product, and return the product.

The test cases are generated so that the answer will fit in a 32-bit integer.

A subarray is a contiguous part of an array.

**Example:**
```
Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.

Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
```

### Golang Solution

```go
func maxProduct(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    
    maxProd := nums[0]
    minProd := nums[0]
    result := nums[0]
    
    for i := 1; i < len(nums); i++ {
        if nums[i] < 0 {
            maxProd, minProd = minProd, maxProd
        }
        
        maxProd = max(nums[i], maxProd*nums[i])
        minProd = min(nums[i], minProd*nums[i])
        
        result = max(result, maxProd)
    }
    
    return result
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

### Alternative Solutions

#### **Brute Force**
```go
func maxProductBruteForce(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    
    maxProd := nums[0]
    
    for i := 0; i < len(nums); i++ {
        product := 1
        for j := i; j < len(nums); j++ {
            product *= nums[j]
            maxProd = max(maxProd, product)
        }
    }
    
    return maxProd
}
```

#### **Two Pass Approach**
```go
func maxProductTwoPass(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    
    maxProd := nums[0]
    product := 1
    
    // Forward pass
    for i := 0; i < len(nums); i++ {
        product *= nums[i]
        maxProd = max(maxProd, product)
        if product == 0 {
            product = 1
        }
    }
    
    product = 1
    
    // Backward pass
    for i := len(nums) - 1; i >= 0; i-- {
        product *= nums[i]
        maxProd = max(maxProd, product)
        if product == 0 {
            product = 1
        }
    }
    
    return maxProd
}
```

#### **Dynamic Programming with Array**
```go
func maxProductDP(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    
    n := len(nums)
    maxDP := make([]int, n)
    minDP := make([]int, n)
    
    maxDP[0] = nums[0]
    minDP[0] = nums[0]
    result := nums[0]
    
    for i := 1; i < n; i++ {
        if nums[i] >= 0 {
            maxDP[i] = max(nums[i], maxDP[i-1]*nums[i])
            minDP[i] = min(nums[i], minDP[i-1]*nums[i])
        } else {
            maxDP[i] = max(nums[i], minDP[i-1]*nums[i])
            minDP[i] = min(nums[i], maxDP[i-1]*nums[i])
        }
        
        result = max(result, maxDP[i])
    }
    
    return result
}
```

#### **Return Subarray Indices**
```go
func maxProductWithIndices(nums []int) (int, int, int) {
    if len(nums) == 0 {
        return 0, -1, -1
    }
    
    maxProd := nums[0]
    minProd := nums[0]
    result := nums[0]
    start, end := 0, 0
    tempStart := 0
    
    for i := 1; i < len(nums); i++ {
        if nums[i] < 0 {
            maxProd, minProd = minProd, maxProd
        }
        
        if nums[i] > maxProd*nums[i] {
            maxProd = nums[i]
            tempStart = i
        } else {
            maxProd = maxProd * nums[i]
        }
        
        if maxProd > result {
            result = maxProd
            start = tempStart
            end = i
        }
    }
    
    return result, start, end
}
```

### Complexity
- **Time Complexity:** O(n) for optimal, O(n²) for brute force
- **Space Complexity:** O(1) for optimal, O(n) for DP array
