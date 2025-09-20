# Sort Colors

### Problem
Given an array `nums` with `n` objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.

We will use the integers `0`, `1`, and `2` to represent the color red, white, and blue, respectively.

You must solve this problem without using the library's sort function.

**Example:**
```
Input: nums = [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]

Input: nums = [2,0,1]
Output: [0,1,2]
```

### Golang Solution

```go
func sortColors(nums []int) {
    left, right := 0, len(nums)-1
    current := 0
    
    for current <= right {
        if nums[current] == 0 {
            nums[left], nums[current] = nums[current], nums[left]
            left++
            current++
        } else if nums[current] == 2 {
            nums[right], nums[current] = nums[current], nums[right]
            right--
        } else {
            current++
        }
    }
}
```

### Alternative Solutions

#### **Counting Sort**
```go
func sortColorsCounting(nums []int) {
    count := make([]int, 3)
    
    for _, num := range nums {
        count[num]++
    }
    
    index := 0
    for color := 0; color < 3; color++ {
        for count[color] > 0 {
            nums[index] = color
            index++
            count[color]--
        }
    }
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)


---

## AUTO-GENERATED: Starter Content

<!-- AUTO-GENERATED - REVIEW REQUIRED -->

This section seeds the document with a short introduction, learning objectives, and related links to code samples.

**Learning objectives:**
- Understand the core concepts.
- See practical code examples.

**Related files:**

Please replace this auto-generated section with curated content.
