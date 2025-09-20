# 🔢 4Sum - LeetCode Problem 18

## Problem Statement

Given an array `nums` of `n` integers, return an array of all the unique quadruplets `[nums[a], nums[b], nums[c], nums[d]]` such that:

- `0 <= a, b, c, d < n`
- `a`, `b`, `c`, and `d` are **distinct**
- `nums[a] + nums[b] + nums[c] + nums[d] == target`

You may return the answer in **any order**.

## Examples

```javascript
// Example 1
Input: nums = [1,0,-1,0,-2,2], target = 0
Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]

// Example 2
Input: nums = [2,2,2,2,2], target = 8
Output: [[2,2,2,2]]
```

## Approach

### Two Pointers Approach (Optimized)

1. **Sort the array** to enable two-pointer technique
2. **Fix first two elements** using nested loops
3. **Use two pointers** for the remaining two elements
4. **Skip duplicates** to avoid duplicate quadruplets
5. **Early termination** for optimization

### Time Complexity
- **Time**: O(n³) - Three nested loops
- **Space**: O(1) - Excluding output array

## Solution

```javascript
/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number[][]}
 */
function fourSum(nums, target) {
  const result = [];
  const n = nums.length;
  
  // Sort array to enable two-pointer technique
  nums.sort((a, b) => a - b);
  
  // Fix first element
  for (let i = 0; i < n - 3; i++) {
    // Skip duplicates for first element
    if (i > 0 && nums[i] === nums[i - 1]) continue;
    
    // Early termination if smallest possible sum > target
    if (nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target) break;
    
    // Early termination if largest possible sum < target
    if (nums[i] + nums[n - 3] + nums[n - 2] + nums[n - 1] < target) continue;
    
    // Fix second element
    for (let j = i + 1; j < n - 2; j++) {
      // Skip duplicates for second element
      if (j > i + 1 && nums[j] === nums[j - 1]) continue;
      
      // Early termination for second element
      if (nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target) break;
      if (nums[i] + nums[j] + nums[n - 2] + nums[n - 1] < target) continue;
      
      // Two pointers for remaining two elements
      let left = j + 1;
      let right = n - 1;
      
      while (left < right) {
        const sum = nums[i] + nums[j] + nums[left] + nums[right];
        
        if (sum === target) {
          result.push([nums[i], nums[j], nums[left], nums[right]]);
          
          // Skip duplicates for left pointer
          while (left < right && nums[left] === nums[left + 1]) left++;
          // Skip duplicates for right pointer
          while (left < right && nums[right] === nums[right - 1]) right--;
          
          left++;
          right--;
        } else if (sum < target) {
          left++;
        } else {
          right--;
        }
      }
    }
  }
  
  return result;
}
```

## Alternative Approach: Hash Set

```javascript
/**
 * Hash Set approach for 4Sum
 * @param {number[]} nums
 * @param {number} target
 * @return {number[][]}
 */
function fourSumHashSet(nums, target) {
  const result = [];
  const n = nums.length;
  const seen = new Set();
  
  nums.sort((a, b) => a - b);
  
  for (let i = 0; i < n - 3; i++) {
    if (i > 0 && nums[i] === nums[i - 1]) continue;
    
    for (let j = i + 1; j < n - 2; j++) {
      if (j > i + 1 && nums[j] === nums[j - 1]) continue;
      
      const twoSum = nums[i] + nums[j];
      const hashSet = new Set();
      
      for (let k = j + 1; k < n; k++) {
        const complement = target - twoSum - nums[k];
        
        if (hashSet.has(complement)) {
          const quadruplet = [nums[i], nums[j], complement, nums[k]];
          const key = quadruplet.join(',');
          
          if (!seen.has(key)) {
            result.push(quadruplet);
            seen.add(key);
          }
        }
        
        hashSet.add(nums[k]);
      }
    }
  }
  
  return result;
}
```

## Generic K-Sum Solution

```javascript
/**
 * Generic K-Sum solution that can handle any k
 * @param {number[]} nums
 * @param {number} target
 * @param {number} k
 * @return {number[][]}
 */
function kSum(nums, target, k) {
  const result = [];
  const n = nums.length;
  
  // Base cases
  if (n < k || k < 2) return result;
  if (k === 2) return twoSum(nums, target);
  
  nums.sort((a, b) => a - b);
  
  for (let i = 0; i < n - k + 1; i++) {
    if (i > 0 && nums[i] === nums[i - 1]) continue;
    
    // Early termination
    if (nums[i] * k > target || nums[n - 1] * k < target) break;
    
    const subResult = kSum(nums.slice(i + 1), target - nums[i], k - 1);
    
    for (const arr of subResult) {
      result.push([nums[i], ...arr]);
    }
  }
  
  return result;
}

/**
 * Helper function for 2Sum
 * @param {number[]} nums
 * @param {number} target
 * @return {number[][]}
 */
function twoSum(nums, target) {
  const result = [];
  let left = 0;
  let right = nums.length - 1;
  
  while (left < right) {
    const sum = nums[left] + nums[right];
    
    if (sum === target) {
      result.push([nums[left], nums[right]]);
      
      while (left < right && nums[left] === nums[left + 1]) left++;
      while (left < right && nums[right] === nums[right - 1]) right--;
      
      left++;
      right--;
    } else if (sum < target) {
      left++;
    } else {
      right--;
    }
  }
  
  return result;
}

// Usage for 4Sum
function fourSumGeneric(nums, target) {
  return kSum(nums, target, 4);
}
```

## Test Cases

```javascript
// Test cases
console.log("=== 4Sum Test Cases ===");

// Test 1
console.log("Test 1:");
console.log("Input: [1,0,-1,0,-2,2], target = 0");
console.log("Output:", fourSum([1,0,-1,0,-2,2], 0));
console.log("Expected: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]");
console.log();

// Test 2
console.log("Test 2:");
console.log("Input: [2,2,2,2,2], target = 8");
console.log("Output:", fourSum([2,2,2,2,2], 8));
console.log("Expected: [[2,2,2,2]]");
console.log();

// Test 3
console.log("Test 3:");
console.log("Input: [1,0,-1,0,-2,2], target = 0");
console.log("Output (HashSet):", fourSumHashSet([1,0,-1,0,-2,2], 0));
console.log();

// Test 4
console.log("Test 4:");
console.log("Input: [1,0,-1,0,-2,2], target = 0");
console.log("Output (Generic):", fourSumGeneric([1,0,-1,0,-2,2], 0));
console.log();

// Performance test
console.log("=== Performance Test ===");
const largeArray = Array.from({length: 100}, () => Math.floor(Math.random() * 100) - 50);
const start = performance.now();
const result = fourSum(largeArray, 0);
const end = performance.now();
console.log(`Time taken: ${end - start}ms`);
console.log(`Found ${result.length} quadruplets`);
```

## Key Insights

1. **Sorting is crucial** for the two-pointer technique
2. **Skip duplicates** at each level to avoid duplicate results
3. **Early termination** optimizations can significantly improve performance
4. **Generic K-Sum** approach can be extended to any k value
5. **Hash set approach** is an alternative but may use more memory

## Common Mistakes

1. **Not sorting the array** before applying two-pointer technique
2. **Not skipping duplicates** leading to duplicate quadruplets
3. **Incorrect pointer movement** in the two-pointer loop
4. **Missing early termination** optimizations
5. **Off-by-one errors** in loop boundaries

## Related Problems

- [Two Sum](TwoSum.md)
- [3Sum](ThreeSum.md)
- [3Sum Closest](../../../algorithms/Arrays/ThreeSumClosest.md)
- [4Sum II](4SumII.md)

## Interview Tips

1. **Start with brute force** and then optimize
2. **Explain the sorting step** and why it's necessary
3. **Discuss time/space complexity** trade-offs
4. **Mention early termination** optimizations
5. **Consider edge cases** like empty arrays or insufficient elements
