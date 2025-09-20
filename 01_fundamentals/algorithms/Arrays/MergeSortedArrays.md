# Merge Sorted Array

### Problem
You are given two integer arrays `nums1` and `nums2`, sorted in non-decreasing order, and two integers `m` and `n`, representing the number of elements in `nums1` and `nums2` respectively.

Merge `nums2` into `nums1` as one sorted array.

**Example:**
```
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
```

### Golang Solution

```go
func merge(nums1 []int, m int, nums2 []int, n int) {
    i, j, k := m-1, n-1, m+n-1
    
    for i >= 0 && j >= 0 {
        if nums1[i] > nums2[j] {
            nums1[k] = nums1[i]
            i--
        } else {
            nums1[k] = nums2[j]
            j--
        }
        k--
    }
    
    // Copy remaining elements from nums2
    for j >= 0 {
        nums1[k] = nums2[j]
        j--
        k--
    }
}
```

### Complexity
- **Time Complexity:** O(m + n)
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
