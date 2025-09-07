# Single Number

### Problem
Given a non-empty array of integers `nums`, every element appears twice except for one. Find that single one.

You must implement a solution with a linear runtime complexity and use only constant extra space.

**Example:**
```
Input: nums = [2,2,1]
Output: 1

Input: nums = [4,1,2,1,2]
Output: 4

Input: nums = [1]
Output: 1
```

**Constraints:**
- 1 ≤ nums.length ≤ 3 × 10⁴
- -3 × 10⁴ ≤ nums[i] ≤ 3 × 10⁴
- Each element in the array appears twice except for one element which appears only once

### Explanation

#### **XOR Approach**
- Use XOR properties: `a ^ a = 0`, `a ^ 0 = a`
- XOR all elements together
- Elements that appear twice will cancel out (XOR with themselves = 0)
- The single element will remain
- Time Complexity: O(n)
- Space Complexity: O(1)

#### **Hash Map Approach**
- Use hash map to count frequency of each element
- Find element with frequency 1
- Time Complexity: O(n)
- Space Complexity: O(n)

#### **Mathematical Approach**
- Calculate sum of all unique elements × 2
- Subtract sum of all elements
- Result is the single element
- Time Complexity: O(n)
- Space Complexity: O(n)

### Dry Run

**Input:** `nums = [4,1,2,1,2]`

#### **XOR Approach**

| Step | Element | XOR Result | Binary |
|------|---------|------------|---------|
| 0 | - | 0 | 0000 |
| 1 | 4 | 0 ^ 4 = 4 | 0100 |
| 2 | 1 | 4 ^ 1 = 5 | 0101 |
| 3 | 2 | 5 ^ 2 = 7 | 0111 |
| 4 | 1 | 7 ^ 1 = 6 | 0110 |
| 5 | 2 | 6 ^ 2 = 4 | 0100 |

**Result:** `4`

### Complexity
- **Time Complexity:** O(n) - Single pass through the array
- **Space Complexity:** O(1) - Only using constant extra space

### Golang Solution

#### **XOR Solution (Optimal)**
```go
func singleNumber(nums []int) int {
    result := 0
    
    for _, num := range nums {
        result ^= num
    }
    
    return result
}
```

#### **Hash Map Solution**
```go
func singleNumberHashMap(nums []int) int {
    freq := make(map[int]int)
    
    // Count frequency of each element
    for _, num := range nums {
        freq[num]++
    }
    
    // Find element with frequency 1
    for num, count := range freq {
        if count == 1 {
            return num
        }
    }
    
    return -1 // Should never reach here
}
```

#### **Mathematical Solution**
```go
func singleNumberMath(nums []int) int {
    sum := 0
    uniqueSum := 0
    seen := make(map[int]bool)
    
    for _, num := range nums {
        sum += num
        if !seen[num] {
            uniqueSum += num
            seen[num] = true
        }
    }
    
    // 2 * uniqueSum - sum = single element
    return 2*uniqueSum - sum
}
```

### Alternative Solutions

#### **Using Sort**
```go
func singleNumberSort(nums []int) int {
    sort.Ints(nums)
    
    for i := 0; i < len(nums)-1; i += 2 {
        if nums[i] != nums[i+1] {
            return nums[i]
        }
    }
    
    // If we reach here, the single element is the last one
    return nums[len(nums)-1]
}
```

#### **Using Set**
```go
func singleNumberSet(nums []int) int {
    seen := make(map[int]bool)
    
    for _, num := range nums {
        if seen[num] {
            delete(seen, num)
        } else {
            seen[num] = true
        }
    }
    
    // Return the only element in the set
    for num := range seen {
        return num
    }
    
    return -1
}
```

### Notes / Variations

#### **Related Problems**
- **Single Number II**: Every element appears three times except one
- **Single Number III**: Two elements appear once, rest appear twice
- **Missing Number**: Find missing number in array
- **Find the Difference**: Find added character in string
- **Find the Duplicate Number**: Find duplicate number in array

#### **ICPC Insights**
- **XOR Properties**: Master XOR properties for bit manipulation
- **Space Efficiency**: XOR approach uses O(1) space
- **Bit Manipulation**: Understand bitwise operations
- **Mathematical Properties**: Use mathematical properties for optimization

#### **Go-Specific Optimizations**
```go
// Use range for cleaner code
func singleNumber(nums []int) int {
    result := 0
    for _, num := range nums {
        result ^= num
    }
    return result
}

// Use pointer to avoid array bounds checking
func singleNumber(nums []int) int {
    if len(nums) == 1 {
        return nums[0]
    }
    
    result := nums[0]
    for i := 1; i < len(nums); i++ {
        result ^= nums[i]
    }
    return result
}
```

#### **Real-World Applications**
- **Error Detection**: XOR for parity checking
- **Cryptography**: XOR for simple encryption
- **Data Compression**: XOR for delta encoding
- **Network Protocols**: XOR for checksums

### Testing

```go
func TestSingleNumber(t *testing.T) {
    tests := []struct {
        nums     []int
        expected int
    }{
        {[]int{2, 2, 1}, 1},
        {[]int{4, 1, 2, 1, 2}, 4},
        {[]int{1}, 1},
        {[]int{1, 1, 2, 2, 3}, 3},
        {[]int{-1, -1, -2}, -2},
    }
    
    for _, test := range tests {
        result := singleNumber(test.nums)
        if result != test.expected {
            t.Errorf("singleNumber(%v) = %d, expected %d", 
                test.nums, result, test.expected)
        }
    }
}
```

### Visualization

```
Input: [4, 1, 2, 1, 2]

XOR Operation:
4 ^ 1 ^ 2 ^ 1 ^ 2

Step by step:
4 ^ 1 = 5 (0100 ^ 0001 = 0101)
5 ^ 2 = 7 (0101 ^ 0010 = 0111)
7 ^ 1 = 6 (0111 ^ 0001 = 0110)
6 ^ 2 = 4 (0110 ^ 0010 = 0100)

Result: 4

Why XOR works:
- 1 ^ 1 = 0 (cancels out)
- 2 ^ 2 = 0 (cancels out)
- 4 ^ 0 = 4 (remains)
```

### Performance Comparison

| Approach | Time | Space | Pros | Cons |
|----------|------|-------|------|------|
| XOR | O(n) | O(1) | Space efficient | Requires XOR knowledge |
| Hash Map | O(n) | O(n) | Easy to understand | Extra space needed |
| Math | O(n) | O(n) | Mathematical approach | Extra space needed |
| Sort | O(n log n) | O(1) | Simple logic | Slower time complexity |

**Recommendation**: Use XOR approach for optimal performance, hash map for clarity.
