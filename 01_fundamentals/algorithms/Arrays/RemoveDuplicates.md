---
# Auto-generated front matter
Title: Removeduplicates
LastUpdated: 2025-11-06T20:45:58.723363
Tags: []
Status: draft
---

# Remove Duplicates from Sorted Array

### Problem
Given an integer array `nums` sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. The relative order of the elements should be kept the same.

Return `k` after placing the final result in the first `k` slots of `nums`.

**Example:**
```
Input: nums = [1,1,2]
Output: 2, nums = [1,2,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 1 and 2 respectively.

Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
```

**Constraints:**
- 1 ≤ nums.length ≤ 3 × 10⁴
- -100 ≤ nums[i] ≤ 100
- nums is sorted in non-decreasing order

### Explanation

#### **Two Pointers Approach**
- Use slow pointer to track position for next unique element
- Use fast pointer to scan through the array
- When we find a new unique element, place it at slow pointer position
- Time Complexity: O(n)
- Space Complexity: O(1)

### Dry Run

**Input:** `nums = [0,0,1,1,1,2,2,3,3,4]`

| Step | slow | fast | nums[fast] | Action |
|------|------|------|------------|---------|
| 0 | 0 | 0 | 0 | Initialize |
| 1 | 1 | 1 | 0 | Skip duplicate |
| 2 | 1 | 2 | 1 | Place 1 at slow, slow++ |
| 3 | 2 | 3 | 1 | Skip duplicate |
| 4 | 2 | 4 | 1 | Skip duplicate |
| 5 | 2 | 5 | 2 | Place 2 at slow, slow++ |
| 6 | 3 | 6 | 2 | Skip duplicate |
| 7 | 3 | 7 | 3 | Place 3 at slow, slow++ |
| 8 | 4 | 8 | 3 | Skip duplicate |
| 9 | 4 | 9 | 4 | Place 4 at slow, slow++ |

**Result:** `k = 5`, `nums = [0,1,2,3,4,_,_,_,_,_]`

### Complexity
- **Time Complexity:** O(n) - Single pass through the array
- **Space Complexity:** O(1) - Only using constant extra space

### Golang Solution

```go
func removeDuplicates(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    
    slow := 0
    
    for fast := 1; fast < len(nums); fast++ {
        if nums[fast] != nums[slow] {
            slow++
            nums[slow] = nums[fast]
        }
    }
    
    return slow + 1
}
```

### Notes / Variations

#### **Related Problems**
- **Remove Duplicates from Sorted Array II**: Allow at most 2 duplicates
- **Remove Element**: Remove all instances of a value
- **Move Zeroes**: Move all zeros to the end
- **Remove Duplicates from Sorted List**: Remove duplicates from linked list

#### **ICPC Insights**
- **In-place Modification**: Modify array without extra space
- **Two Pointers**: Classic technique for array manipulation
- **Sorted Array**: Leverage sorted property for efficiency
- **Return Value**: Return new length, not the array
