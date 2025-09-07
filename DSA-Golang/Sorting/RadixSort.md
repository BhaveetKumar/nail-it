# Radix Sort

### Problem
Implement Radix Sort algorithm to sort an array of integers.

Radix Sort is a non-comparison based sorting algorithm that sorts data with integer keys by grouping keys by the individual digits which share the same significant position and value.

**Example:**
```
Input: [170, 45, 75, 90, 2, 802, 24, 66]
Output: [2, 24, 45, 66, 75, 90, 170, 802]
```

### Golang Solution

```go
func radixSort(nums []int) []int {
    if len(nums) == 0 {
        return nums
    }
    
    // Find the maximum number to know number of digits
    max := nums[0]
    for _, num := range nums {
        if num > max {
            max = num
        }
    }
    
    // Do counting sort for every digit
    for exp := 1; max/exp > 0; exp *= 10 {
        countingSortByDigit(nums, exp)
    }
    
    return nums
}

func countingSortByDigit(nums []int, exp int) {
    n := len(nums)
    output := make([]int, n)
    count := make([]int, 10)
    
    // Store count of occurrences in count[]
    for i := 0; i < n; i++ {
        count[(nums[i]/exp)%10]++
    }
    
    // Change count[i] so that count[i] now contains actual
    // position of this digit in output[]
    for i := 1; i < 10; i++ {
        count[i] += count[i-1]
    }
    
    // Build the output array
    for i := n - 1; i >= 0; i-- {
        output[count[(nums[i]/exp)%10]-1] = nums[i]
        count[(nums[i]/exp)%10]--
    }
    
    // Copy the output array to nums[], so that nums[] now
    // contains sorted numbers according to current digit
    for i := 0; i < n; i++ {
        nums[i] = output[i]
    }
}
```

### Alternative Solutions

#### **Using Buckets**
```go
func radixSortBuckets(nums []int) []int {
    if len(nums) == 0 {
        return nums
    }
    
    // Find maximum number
    max := nums[0]
    for _, num := range nums {
        if num > max {
            max = num
        }
    }
    
    // Do bucket sort for every digit
    for exp := 1; max/exp > 0; exp *= 10 {
        bucketSortByDigit(nums, exp)
    }
    
    return nums
}

func bucketSortByDigit(nums []int, exp int) {
    buckets := make([][]int, 10)
    
    // Distribute numbers into buckets
    for _, num := range nums {
        digit := (num / exp) % 10
        buckets[digit] = append(buckets[digit], num)
    }
    
    // Collect numbers from buckets
    index := 0
    for i := 0; i < 10; i++ {
        for _, num := range buckets[i] {
            nums[index] = num
            index++
        }
    }
}
```

#### **For Negative Numbers**
```go
func radixSortWithNegatives(nums []int) []int {
    if len(nums) == 0 {
        return nums
    }
    
    // Separate positive and negative numbers
    var positives, negatives []int
    for _, num := range nums {
        if num >= 0 {
            positives = append(positives, num)
        } else {
            negatives = append(negatives, -num) // Make positive for sorting
        }
    }
    
    // Sort positive numbers
    if len(positives) > 0 {
        radixSort(positives)
    }
    
    // Sort negative numbers (reversed)
    if len(negatives) > 0 {
        radixSort(negatives)
        // Reverse and make negative again
        for i, j := 0, len(negatives)-1; i < j; i, j = i+1, j-1 {
            negatives[i], negatives[j] = -negatives[j], -negatives[i]
        }
    }
    
    // Combine results
    result := make([]int, 0, len(nums))
    result = append(result, negatives...)
    result = append(result, positives...)
    
    return result
}
```

### Complexity
- **Time Complexity:** O(d × (n + k)) where d is number of digits, k is range of digits
- **Space Complexity:** O(n + k)
