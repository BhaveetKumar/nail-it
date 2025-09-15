# Next Greater Element

### Problem
The next greater element of some element `x` in an array is the first greater element that is to the right of `x` in the same array.

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
    nextGreater := make(map[int]int)
    stack := []int{}
    
    // Build next greater element map for nums2
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
        if val, exists := nextGreater[num]; exists {
            result[i] = val
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
        // Find index of num1 in nums2
        index := -1
        for j, num2 := range nums2 {
            if num2 == num1 {
                index = j
                break
            }
        }
        
        // Find next greater element
        result[i] = -1
        for j := index + 1; j < len(nums2); j++ {
            if nums2[j] > num1 {
                result[i] = nums2[j]
                break
            }
        }
    }
    
    return result
}
```

#### **Using Hash Map**
```go
func nextGreaterElementHashMap(nums1 []int, nums2 []int) []int {
    // Create index map for nums2
    indexMap := make(map[int]int)
    for i, num := range nums2 {
        indexMap[num] = i
    }
    
    result := make([]int, len(nums1))
    
    for i, num1 := range nums1 {
        startIndex := indexMap[num1]
        result[i] = -1
        
        for j := startIndex + 1; j < len(nums2); j++ {
            if nums2[j] > num1 {
                result[i] = nums2[j]
                break
            }
        }
    }
    
    return result
}
```

#### **Return with Indices**
```go
type NextGreaterResult struct {
    Value int
    Index int
}

func nextGreaterElementWithIndices(nums1 []int, nums2 []int) []NextGreaterResult {
    nextGreater := make(map[int]NextGreaterResult)
    stack := []int{}
    
    // Build next greater element map for nums2
    for i, num := range nums2 {
        for len(stack) > 0 && nums2[stack[len(stack)-1]] < num {
            prevIndex := stack[len(stack)-1]
            nextGreater[nums2[prevIndex]] = NextGreaterResult{
                Value: num,
                Index: i,
            }
            stack = stack[:len(stack)-1]
        }
        stack = append(stack, i)
    }
    
    // Build result for nums1
    result := make([]NextGreaterResult, len(nums1))
    for i, num := range nums1 {
        if val, exists := nextGreater[num]; exists {
            result[i] = val
        } else {
            result[i] = NextGreaterResult{Value: -1, Index: -1}
        }
    }
    
    return result
}
```

#### **Next Greater Element in Circular Array**
```go
func nextGreaterElementsCircular(nums []int) []int {
    n := len(nums)
    result := make([]int, n)
    stack := []int{}
    
    // Initialize result with -1
    for i := range result {
        result[i] = -1
    }
    
    // Process array twice to handle circular nature
    for i := 0; i < 2*n; i++ {
        num := nums[i%n]
        
        for len(stack) > 0 && nums[stack[len(stack)-1]] < num {
            result[stack[len(stack)-1]] = num
            stack = stack[:len(stack)-1]
        }
        
        if i < n {
            stack = append(stack, i)
        }
    }
    
    return result
}
```

#### **Return All Next Greater Elements**
```go
func nextGreaterElementsAll(nums []int) [][]int {
    n := len(nums)
    result := make([][]int, n)
    
    for i := 0; i < n; i++ {
        var greater []int
        for j := i + 1; j < n; j++ {
            if nums[j] > nums[i] {
                greater = append(greater, nums[j])
            }
        }
        result[i] = greater
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n + m) where n is length of nums1 and m is length of nums2
- **Space Complexity:** O(m) for the stack and hash map