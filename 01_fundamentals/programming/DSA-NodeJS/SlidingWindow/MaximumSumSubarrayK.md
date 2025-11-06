---
# Auto-generated front matter
Title: Maximumsumsubarrayk
LastUpdated: 2025-11-06T20:45:58.787581
Tags: []
Status: draft
---

# Maximum Sum Subarray of Size K

## Problem Statement

Given an array of integers and a number k, find the maximum sum of any contiguous subarray of size k.

**Example 1:**
```
Input: arr = [2, 1, 5, 1, 3, 2], k = 3
Output: 9
Explanation: Subarray with maximum sum is [5, 1, 3] with sum = 9
```

**Example 2:**
```
Input: arr = [2, 3, 4, 1, 5], k = 2
Output: 7
Explanation: Subarray with maximum sum is [3, 4] with sum = 7
```

## Approach

### Brute Force Approach
1. Generate all possible subarrays of size k
2. Calculate sum for each subarray
3. Return the maximum sum

**Time Complexity:** O(n × k) - For each position, calculate sum of k elements
**Space Complexity:** O(1) - Only store maximum sum

### Sliding Window Approach
1. Calculate sum of first k elements
2. Slide the window by removing leftmost element and adding rightmost element
3. Keep track of maximum sum encountered

**Time Complexity:** O(n) - Single pass through array
**Space Complexity:** O(1) - Only store window sum and maximum

## Solution

```javascript
/**
 * Find maximum sum of subarray of size k using sliding window
 * @param {number[]} arr - Array of integers
 * @param {number} k - Size of subarray
 * @return {number} - Maximum sum of subarray of size k
 */
function maxSumSubarrayK(arr, k) {
    if (arr.length < k) {
        throw new Error("Array length must be at least k");
    }
    
    // Calculate sum of first window
    let windowSum = 0;
    for (let i = 0; i < k; i++) {
        windowSum += arr[i];
    }
    
    let maxSum = windowSum;
    
    // Slide the window
    for (let i = k; i < arr.length; i++) {
        // Remove leftmost element and add rightmost element
        windowSum = windowSum - arr[i - k] + arr[i];
        maxSum = Math.max(maxSum, windowSum);
    }
    
    return maxSum;
}

// Alternative implementation with explicit left and right pointers
function maxSumSubarrayKPointers(arr, k) {
    if (arr.length < k) {
        throw new Error("Array length must be at least k");
    }
    
    let left = 0;
    let windowSum = 0;
    let maxSum = -Infinity;
    
    // Expand window to size k
    for (let right = 0; right < arr.length; right++) {
        windowSum += arr[right];
        
        // When window reaches size k, start sliding
        if (right - left + 1 === k) {
            maxSum = Math.max(maxSum, windowSum);
            windowSum -= arr[left];
            left++;
        }
    }
    
    return maxSum;
}

// Return both maximum sum and the subarray
function maxSumSubarrayKWithIndices(arr, k) {
    if (arr.length < k) {
        throw new Error("Array length must be at least k");
    }
    
    // Calculate sum of first window
    let windowSum = 0;
    for (let i = 0; i < k; i++) {
        windowSum += arr[i];
    }
    
    let maxSum = windowSum;
    let maxStart = 0;
    
    // Slide the window
    for (let i = k; i < arr.length; i++) {
        windowSum = windowSum - arr[i - k] + arr[i];
        
        if (windowSum > maxSum) {
            maxSum = windowSum;
            maxStart = i - k + 1;
        }
    }
    
    return {
        maxSum,
        subarray: arr.slice(maxStart, maxStart + k),
        startIndex: maxStart,
        endIndex: maxStart + k - 1
    };
}
```

## Dry Run

**Input:** arr = [2, 1, 5, 1, 3, 2], k = 3

```
Initial: windowSum = 0, maxSum = -∞

Step 1: Calculate sum of first window [2, 1, 5]
        windowSum = 2 + 1 + 5 = 8
        maxSum = max(-∞, 8) = 8

Step 2: Slide window to [1, 5, 1]
        windowSum = 8 - 2 + 1 = 7
        maxSum = max(8, 7) = 8

Step 3: Slide window to [5, 1, 3]
        windowSum = 7 - 1 + 3 = 9
        maxSum = max(8, 9) = 9

Step 4: Slide window to [1, 3, 2]
        windowSum = 9 - 5 + 2 = 6
        maxSum = max(9, 6) = 9

Result: maxSum = 9
```

## Complexity Analysis

- **Time Complexity:** O(n) - Single pass through array
- **Space Complexity:** O(1) - Only store window sum and maximum

## Alternative Solutions

### Brute Force Approach
```javascript
function maxSumSubarrayKBruteForce(arr, k) {
    if (arr.length < k) {
        throw new Error("Array length must be at least k");
    }
    
    let maxSum = -Infinity;
    
    for (let i = 0; i <= arr.length - k; i++) {
        let currentSum = 0;
        for (let j = i; j < i + k; j++) {
            currentSum += arr[j];
        }
        maxSum = Math.max(maxSum, currentSum);
    }
    
    return maxSum;
}
```

### Using Prefix Sum
```javascript
function maxSumSubarrayKPrefixSum(arr, k) {
    if (arr.length < k) {
        throw new Error("Array length must be at least k");
    }
    
    // Calculate prefix sum
    const prefixSum = [0];
    for (let i = 0; i < arr.length; i++) {
        prefixSum[i + 1] = prefixSum[i] + arr[i];
    }
    
    let maxSum = -Infinity;
    
    // Find maximum sum of subarray of size k
    for (let i = 0; i <= arr.length - k; i++) {
        const currentSum = prefixSum[i + k] - prefixSum[i];
        maxSum = Math.max(maxSum, currentSum);
    }
    
    return maxSum;
}
```

### Generic Sliding Window Template
```javascript
function slidingWindowTemplate(arr, k) {
    let left = 0;
    let result = 0;
    let windowSum = 0;
    
    // Expand window
    for (let right = 0; right < arr.length; right++) {
        windowSum += arr[right];
        
        // Shrink window if needed
        while (right - left + 1 > k) {
            windowSum -= arr[left];
            left++;
        }
        
        // Process window
        if (right - left + 1 === k) {
            result = Math.max(result, windowSum);
        }
    }
    
    return result;
}
```

## Test Cases

```javascript
// Test cases
console.log(maxSumSubarrayK([2, 1, 5, 1, 3, 2], 3)); // 9
console.log(maxSumSubarrayK([2, 3, 4, 1, 5], 2)); // 7
console.log(maxSumSubarrayK([1, 4, 2, 10, 23, 3, 1, 0, 20], 4)); // 39
console.log(maxSumSubarrayK([100, 200, 300, 400], 2)); // 700

// Edge cases
console.log(maxSumSubarrayK([1, 2, 3], 3)); // 6 (entire array)
console.log(maxSumSubarrayK([5], 1)); // 5 (single element)

// With indices
console.log(maxSumSubarrayKWithIndices([2, 1, 5, 1, 3, 2], 3));
// { maxSum: 9, subarray: [5, 1, 3], startIndex: 2, endIndex: 4 }

// Error cases
try {
    maxSumSubarrayK([1, 2], 3); // Error: Array length must be at least k
} catch (error) {
    console.log(error.message);
}
```

## Key Insights

1. **Sliding Window**: Efficiently reuse previous calculations
2. **Fixed Window Size**: Window size remains constant
3. **Single Pass**: Process array in O(n) time
4. **Space Efficient**: Only need O(1) extra space
5. **Avoid Recalculation**: Don't recalculate sum from scratch

## Related Problems

- [Maximum Sum Subarray](MaximumSumSubarray.md) - Variable size window
- [Longest Substring Without Repeating Characters](LongestSubstringNoRepeat.md) - Variable size window
- [Minimum Window Substring](../../../algorithms/SlidingWindow/MinimumWindowSubstring.md) - Variable size window
- [Fruit Into Baskets](../../../algorithms/SlidingWindow/FruitIntoBaskets.md) - Variable size window
- [Longest Substring with At Most K Distinct Characters](LongestSubstringKDistinct.md) - Variable size window
