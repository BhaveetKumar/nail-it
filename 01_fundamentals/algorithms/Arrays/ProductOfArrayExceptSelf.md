# Product of Array Except Self

### Problem
Given an integer array `nums`, return an array `answer` such that `answer[i]` is equal to the product of all the elements of `nums` except `nums[i]`.

The product of any prefix or suffix of `nums` is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operator.

**Example:**
```
Input: nums = [1,2,3,4]
Output: [24,12,8,6]

Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]
```

### Golang Solution

```go
func productExceptSelf(nums []int) []int {
    n := len(nums)
    result := make([]int, n)
    
    // Calculate left products
    result[0] = 1
    for i := 1; i < n; i++ {
        result[i] = result[i-1] * nums[i-1]
    }
    
    // Calculate right products and multiply
    rightProduct := 1
    for i := n - 1; i >= 0; i-- {
        result[i] = result[i] * rightProduct
        rightProduct *= nums[i]
    }
    
    return result
}
```

### Alternative Solutions

#### **Using Two Arrays**
```go
func productExceptSelfTwoArrays(nums []int) []int {
    n := len(nums)
    left := make([]int, n)
    right := make([]int, n)
    result := make([]int, n)
    
    // Calculate left products
    left[0] = 1
    for i := 1; i < n; i++ {
        left[i] = left[i-1] * nums[i-1]
    }
    
    // Calculate right products
    right[n-1] = 1
    for i := n - 2; i >= 0; i-- {
        right[i] = right[i+1] * nums[i+1]
    }
    
    // Calculate result
    for i := 0; i < n; i++ {
        result[i] = left[i] * right[i]
    }
    
    return result
}
```

#### **Using Division (Not Recommended)**
```go
func productExceptSelfDivision(nums []int) []int {
    n := len(nums)
    result := make([]int, n)
    totalProduct := 1
    zeroCount := 0
    zeroIndex := -1
    
    // Calculate total product and count zeros
    for i := 0; i < n; i++ {
        if nums[i] == 0 {
            zeroCount++
            zeroIndex = i
        } else {
            totalProduct *= nums[i]
        }
    }
    
    // Handle different cases
    if zeroCount > 1 {
        // All elements are 0
        return result
    } else if zeroCount == 1 {
        // Only the zero element is non-zero
        result[zeroIndex] = totalProduct
        return result
    } else {
        // No zeros
        for i := 0; i < n; i++ {
            result[i] = totalProduct / nums[i]
        }
    }
    
    return result
}
```

#### **Using Prefix and Suffix**
```go
func productExceptSelfPrefixSuffix(nums []int) []int {
    n := len(nums)
    prefix := make([]int, n)
    suffix := make([]int, n)
    result := make([]int, n)
    
    // Calculate prefix products
    prefix[0] = nums[0]
    for i := 1; i < n; i++ {
        prefix[i] = prefix[i-1] * nums[i]
    }
    
    // Calculate suffix products
    suffix[n-1] = nums[n-1]
    for i := n - 2; i >= 0; i-- {
        suffix[i] = suffix[i+1] * nums[i]
    }
    
    // Calculate result
    for i := 0; i < n; i++ {
        if i == 0 {
            result[i] = suffix[i+1]
        } else if i == n-1 {
            result[i] = prefix[i-1]
        } else {
            result[i] = prefix[i-1] * suffix[i+1]
        }
    }
    
    return result
}
```

#### **In-Place with Extra Space**
```go
func productExceptSelfInPlace(nums []int) []int {
    n := len(nums)
    result := make([]int, n)
    
    // First pass: calculate left products
    result[0] = 1
    for i := 1; i < n; i++ {
        result[i] = result[i-1] * nums[i-1]
    }
    
    // Second pass: calculate right products and multiply
    rightProduct := 1
    for i := n - 1; i >= 0; i-- {
        result[i] *= rightProduct
        rightProduct *= nums[i]
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for optimal, O(n) for two arrays