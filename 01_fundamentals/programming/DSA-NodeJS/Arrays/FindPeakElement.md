# 🏔️ Find Peak Element - LeetCode Problem 162

## Problem Statement

A peak element is an element that is strictly greater than its neighbors.

Given a **0-indexed** integer array `nums`, find a peak element, and return its index. If the array contains multiple peaks, return the index to **any of the peaks**.

You may imagine that `nums[-1] = nums[n] = -∞`. In other words, an element is always considered to be strictly greater than a neighbor that is outside the array.

You must write an algorithm that runs in `O(log n)` time.

## Examples

```javascript
// Example 1
Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.

// Example 2
Input: nums = [1,2,1,3,5,6,4]
Output: 5
Explanation: Your function can return either index number 1 where the peak element is 2, or index number 5 where the peak element is 6.
```

## Approach

### Binary Search (Optimal)

Since we need O(log n) time complexity, we use **binary search**:

1. **Compare middle element** with its neighbors
2. **If it's a peak**, return its index
3. **If left neighbor is greater**, search in left half
4. **If right neighbor is greater**, search in right half
5. **Guaranteed to find a peak** due to the boundary conditions

### Time Complexity
- **Time**: O(log n) - Binary search
- **Space**: O(1) - Constant extra space

## Solution

### Approach 1: Binary Search (Iterative)

```javascript
/**
 * @param {number[]} nums
 * @return {number}
 */
function findPeakElement(nums) {
  const n = nums.length;
  
  // Handle edge cases
  if (n === 1) return 0;
  if (nums[0] > nums[1]) return 0;
  if (nums[n - 1] > nums[n - 2]) return n - 1;
  
  let left = 1;
  let right = n - 2;
  
  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    
    // Check if mid is a peak
    if (nums[mid] > nums[mid - 1] && nums[mid] > nums[mid + 1]) {
      return mid;
    }
    
    // If left neighbor is greater, search left
    if (nums[mid - 1] > nums[mid]) {
      right = mid - 1;
    } else {
      // If right neighbor is greater, search right
      left = mid + 1;
    }
  }
  
  return -1; // Should never reach here
}
```

### Approach 2: Binary Search (Recursive)

```javascript
/**
 * Recursive binary search approach
 * @param {number[]} nums
 * @return {number}
 */
function findPeakElementRecursive(nums) {
  function search(left, right) {
    if (left === right) return left;
    
    const mid = Math.floor((left + right) / 2);
    
    if (nums[mid] > nums[mid + 1]) {
      return search(left, mid);
    } else {
      return search(mid + 1, right);
    }
  }
  
  return search(0, nums.length - 1);
}
```

### Approach 3: Linear Search (Brute Force)

```javascript
/**
 * Linear search approach - O(n) time
 * @param {number[]} nums
 * @return {number}
 */
function findPeakElementLinear(nums) {
  const n = nums.length;
  
  for (let i = 0; i < n; i++) {
    const left = i === 0 ? -Infinity : nums[i - 1];
    const right = i === n - 1 ? -Infinity : nums[i + 1];
    
    if (nums[i] > left && nums[i] > right) {
      return i;
    }
  }
  
  return -1;
}
```

### Approach 4: Find All Peaks

```javascript
/**
 * Find all peak elements
 * @param {number[]} nums
 * @return {number[]}
 */
function findAllPeaks(nums) {
  const peaks = [];
  const n = nums.length;
  
  for (let i = 0; i < n; i++) {
    const left = i === 0 ? -Infinity : nums[i - 1];
    const right = i === n - 1 ? -Infinity : nums[i + 1];
    
    if (nums[i] > left && nums[i] > right) {
      peaks.push(i);
    }
  }
  
  return peaks;
}
```

### Approach 5: Find Global Maximum (Alternative)

```javascript
/**
 * Find the global maximum (always a peak)
 * @param {number[]} nums
 * @return {number}
 */
function findPeakElementGlobalMax(nums) {
  let maxIndex = 0;
  
  for (let i = 1; i < nums.length; i++) {
    if (nums[i] > nums[maxIndex]) {
      maxIndex = i;
    }
  }
  
  return maxIndex;
}
```

## Advanced: 2D Peak Finding

```javascript
/**
 * Find peak in 2D array
 * @param {number[][]} matrix
 * @return {number[]}
 */
function find2DPeak(matrix) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  
  function findPeakInRow(row) {
    let left = 0;
    let right = cols - 1;
    
    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      
      if (matrix[row][mid] > matrix[row][mid + 1]) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
    
    return left;
  }
  
  let top = 0;
  let bottom = rows - 1;
  
  while (top < bottom) {
    const mid = Math.floor((top + bottom) / 2);
    const col = findPeakInRow(mid);
    
    if (matrix[mid][col] > matrix[mid + 1][col]) {
      bottom = mid;
    } else {
      top = mid + 1;
    }
  }
  
  const col = findPeakInRow(top);
  return [top, col];
}
```

## Test Cases

```javascript
// Test cases
console.log("=== Find Peak Element Test Cases ===");

// Test 1
console.log("Test 1:");
console.log("Input: [1,2,3,1]");
console.log("Output:", findPeakElement([1,2,3,1]));
console.log("Expected: 2");
console.log();

// Test 2
console.log("Test 2:");
console.log("Input: [1,2,1,3,5,6,4]");
console.log("Output:", findPeakElement([1,2,1,3,5,6,4]));
console.log("Expected: 1 or 5");
console.log();

// Test 3
console.log("Test 3:");
console.log("Input: [1]");
console.log("Output:", findPeakElement([1]));
console.log("Expected: 0");
console.log();

// Test 4
console.log("Test 4:");
console.log("Input: [1,2]");
console.log("Output:", findPeakElement([1,2]));
console.log("Expected: 1");
console.log();

// Test 5
console.log("Test 5:");
console.log("Input: [2,1]");
console.log("Output:", findPeakElement([2,1]));
console.log("Expected: 0");
console.log();

// Test all peaks
console.log("=== Find All Peaks ===");
const testArray = [1,2,1,3,5,6,4];
console.log("Input:", testArray);
console.log("All peaks:", findAllPeaks(testArray));
console.log();

// Performance comparison
console.log("=== Performance Comparison ===");
const largeArray = Array.from({length: 1000000}, (_, i) => Math.sin(i * 0.01) * 100);

console.log("Testing with array of length 1,000,000:");

// Binary search
let start = performance.now();
let result1 = findPeakElement(largeArray);
let end = performance.now();
console.log(`Binary Search: Index ${result1}, Value ${largeArray[result1]} - Time: ${end - start}ms`);

// Linear search
start = performance.now();
let result2 = findPeakElementLinear(largeArray);
end = performance.now();
console.log(`Linear Search: Index ${result2}, Value ${largeArray[result2]} - Time: ${end - start}ms`);

// Global max
start = performance.now();
let result3 = findPeakElementGlobalMax(largeArray);
end = performance.now();
console.log(`Global Max: Index ${result3}, Value ${largeArray[result3]} - Time: ${end - start}ms`);

// 2D Peak test
console.log("\n=== 2D Peak Finding ===");
const matrix2D = [
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16]
];
console.log("2D Matrix:");
matrix2D.forEach(row => console.log(row));
console.log("Peak at:", find2DPeak(matrix2D));
```

## Visualization

```javascript
/**
 * Visualize the array with peak highlighted
 * @param {number[]} nums
 * @param {number} peakIndex
 */
function visualizePeak(nums, peakIndex) {
  console.log("Array visualization:");
  console.log("Index: ", nums.map((_, i) => i.toString().padStart(2)).join(" "));
  console.log("Value: ", nums.map(n => n.toString().padStart(2)).join(" "));
  console.log("Peak:  ", nums.map((_, i) => i === peakIndex ? "^^" : "  ").join(" "));
  console.log(`Peak element: ${nums[peakIndex]} at index ${peakIndex}`);
}

// Example visualization
console.log("=== Visualization ===");
const exampleArray = [1,2,1,3,5,6,4];
const peakIndex = findPeakElement(exampleArray);
visualizePeak(exampleArray, peakIndex);
```

## Key Insights

1. **Binary Search Applicability**: Even though array isn't sorted, we can still use binary search
2. **Boundary Conditions**: Elements at boundaries are considered peaks if they're greater than their single neighbor
3. **Guaranteed Solution**: Due to boundary conditions, there's always at least one peak
4. **Direction Decision**: Compare with neighbors to decide search direction
5. **Multiple Peaks**: Problem asks for any peak, not necessarily the global maximum

## Common Mistakes

1. **Not handling edge cases** (single element, two elements)
2. **Incorrect boundary conditions** in binary search
3. **Off-by-one errors** in array indexing
4. **Not understanding the problem** - any peak is acceptable
5. **Using linear search** when O(log n) is required

## Related Problems

- [Find Minimum in Rotated Sorted Array](../../../algorithms/Arrays/FindMinimumInRotatedSortedArray.md)
- [Search in Rotated Sorted Array](../../../algorithms/Searching/SearchInRotatedSortedArray.md)
- [Mountain Array](MountainArray.md/)
- [Peak Index in Mountain Array](PeakIndexInMountainArray.md/)

## Interview Tips

1. **Start with linear approach** and then optimize
2. **Explain why binary search works** despite unsorted array
3. **Handle edge cases** carefully
4. **Discuss the boundary conditions** and why they guarantee a solution
5. **Mention that any peak is acceptable** - not necessarily the global maximum
6. **Consider the 2D extension** to show deeper understanding
