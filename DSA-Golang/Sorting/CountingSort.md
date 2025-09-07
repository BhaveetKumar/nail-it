# Counting Sort

### Problem
Implement Counting Sort algorithm to sort an array of integers.

Counting Sort is a non-comparison based sorting algorithm that works by counting the number of objects having distinct key values. It is efficient when the range of input data is not significantly greater than the number of objects to be sorted.

**Example:**
```
Input: [4, 2, 2, 8, 3, 3, 1]
Output: [1, 2, 2, 3, 3, 4, 8]
```

### Golang Solution

```go
func countingSort(nums []int) []int {
    if len(nums) == 0 {
        return nums
    }
    
    // Find the range of input data
    min, max := nums[0], nums[0]
    for _, num := range nums {
        if num < min {
            min = num
        }
        if num > max {
            max = num
        }
    }
    
    // Create count array
    rangeSize := max - min + 1
    count := make([]int, rangeSize)
    
    // Count occurrences of each number
    for _, num := range nums {
        count[num-min]++
    }
    
    // Modify count array to store actual position
    for i := 1; i < rangeSize; i++ {
        count[i] += count[i-1]
    }
    
    // Build output array
    output := make([]int, len(nums))
    for i := len(nums) - 1; i >= 0; i-- {
        output[count[nums[i]-min]-1] = nums[i]
        count[nums[i]-min]--
    }
    
    return output
}
```

### Alternative Solutions

#### **In-Place Counting Sort**
```go
func countingSortInPlace(nums []int) {
    if len(nums) == 0 {
        return
    }
    
    // Find the range
    min, max := nums[0], nums[0]
    for _, num := range nums {
        if num < min {
            min = num
        }
        if num > max {
            max = num
        }
    }
    
    // Count occurrences
    rangeSize := max - min + 1
    count := make([]int, rangeSize)
    
    for _, num := range nums {
        count[num-min]++
    }
    
    // Reconstruct the array
    index := 0
    for i := 0; i < rangeSize; i++ {
        for count[i] > 0 {
            nums[index] = i + min
            index++
            count[i]--
        }
    }
}
```

#### **Counting Sort for Strings**
```go
func countingSortString(str string) string {
    if len(str) == 0 {
        return str
    }
    
    // Count occurrences of each character
    count := make([]int, 256) // ASCII characters
    
    for _, char := range str {
        count[char]++
    }
    
    // Reconstruct the string
    result := make([]rune, len(str))
    index := 0
    
    for i := 0; i < 256; i++ {
        for count[i] > 0 {
            result[index] = rune(i)
            index++
            count[i]--
        }
    }
    
    return string(result)
}
```

### Complexity
- **Time Complexity:** O(n + k) where k is the range of input
- **Space Complexity:** O(k)
