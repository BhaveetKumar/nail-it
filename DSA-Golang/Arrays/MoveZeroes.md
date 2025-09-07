# Move Zeroes

### Problem
Given an integer array `nums`, move all `0`'s to the end of it while maintaining the relative order of the non-zero elements.

Note that you must do this in-place without making a copy of the array.

**Example:**
```
Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]

Input: nums = [0]
Output: [0]
```

**Constraints:**
- 1 ≤ nums.length ≤ 10⁴
- -2³¹ ≤ nums[i] ≤ 2³¹ - 1

### Explanation

#### **Two Pointers Approach**
- Use slow pointer to track position for next non-zero element
- Use fast pointer to scan through the array
- When we find a non-zero element, place it at slow pointer position
- Fill remaining positions with zeros
- Time Complexity: O(n)
- Space Complexity: O(1)

### Dry Run

**Input:** `nums = [0,1,0,3,12]`

| Step | slow | fast | nums[fast] | Action |
|------|------|------|------------|---------|
| 0 | 0 | 0 | 0 | Skip zero |
| 1 | 0 | 1 | 1 | Place 1 at slow, slow++ |
| 2 | 1 | 2 | 0 | Skip zero |
| 3 | 1 | 3 | 3 | Place 3 at slow, slow++ |
| 4 | 2 | 4 | 12 | Place 12 at slow, slow++ |

After loop: `nums = [1,3,12,3,12]`
Fill remaining with zeros: `nums = [1,3,12,0,0]`

**Result:** `[1,3,12,0,0]`

### Complexity
- **Time Complexity:** O(n) - Single pass through the array
- **Space Complexity:** O(1) - Only using constant extra space

### Golang Solution

```go
func moveZeroes(nums []int) {
    slow := 0
    
    // Move all non-zero elements to the front
    for fast := 0; fast < len(nums); fast++ {
        if nums[fast] != 0 {
            nums[slow] = nums[fast]
            slow++
        }
    }
    
    // Fill remaining positions with zeros
    for slow < len(nums) {
        nums[slow] = 0
        slow++
    }
}
```

### Alternative Solutions

#### **Swap Approach**
```go
func moveZeroesSwap(nums []int) {
    left := 0
    
    for right := 0; right < len(nums); right++ {
        if nums[right] != 0 {
            nums[left], nums[right] = nums[right], nums[left]
            left++
        }
    }
}
```

### Notes / Variations

#### **Related Problems**
- **Remove Element**: Remove all instances of a value
- **Remove Duplicates**: Remove duplicates from sorted array
- **Sort Colors**: Sort array with three colors
- **Partition Array**: Partition array around a pivot

#### **ICPC Insights**
- **In-place Modification**: Modify array without extra space
- **Two Pointers**: Classic technique for array manipulation
- **Relative Order**: Maintain order of non-zero elements
- **Zero Handling**: Handle zeros efficiently
