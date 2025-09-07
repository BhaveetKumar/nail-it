# Next Greater Element

### Problem
The next greater element of some element x in an array is the first greater element that is to the right of x in the same array.

You are given two distinct 0-indexed integer arrays `nums1` and `nums2`, where `nums1` is a subset of `nums2`.

For each `0 <= i < nums1.length`, find the index `j` such that `nums1[i] == nums2[j]` and determine the next greater element of `nums2[j]` in `nums2`. If there is no next greater element, then the answer for this query is `-1`.

Return an array `ans` of length `nums1.length` such that `ans[i]` is the next greater element as described above.

**Example:**
```
Input: nums1 = [4,1,2], nums2 = [1,3,4,2]
Output: [-1,3,-1]
Explanation: The next greater element for each value of nums1 is as follows:
- 4 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.
- 1 is underlined in nums2 = [1,3,4,2]. The next greater element is 3.
- 2 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.
```

### Golang Solution

```go
func nextGreaterElement(nums1 []int, nums2 []int) []int {
    // Create a map to store next greater element for each number in nums2
    nextGreater := make(map[int]int)
    stack := []int{}
    
    // Find next greater element for each element in nums2
    for _, num := range nums2 {
        for len(stack) > 0 && stack[len(stack)-1] < num {
            nextGreater[stack[len(stack)-1]] = num
            stack = stack[:len(stack)-1]
        }
        stack = append(stack, num)
    }
    
    // Build result for nums1
    result := make([]int, len(nums1))
    for i, num := range nums1 {
        if next, exists := nextGreater[num]; exists {
            result[i] = next
        } else {
            result[i] = -1
        }
    }
    
    return result
}
```

### Alternative Solutions

#### **Brute Force**
```go
func nextGreaterElementBruteForce(nums1 []int, nums2 []int) []int {
    result := make([]int, len(nums1))
    
    for i, num1 := range nums1 {
        // Find the index of num1 in nums2
        j := 0
        for j < len(nums2) && nums2[j] != num1 {
            j++
        }
        
        // Find next greater element
        result[i] = -1
        for k := j + 1; k < len(nums2); k++ {
            if nums2[k] > num1 {
                result[i] = nums2[k]
                break
            }
        }
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(m + n) for stack, O(m × n) for brute force
- **Space Complexity:** O(n)
