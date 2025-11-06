---
# Auto-generated front matter
Title: Twosum
LastUpdated: 2025-11-06T20:45:58.789730
Tags: []
Status: draft
---

# 🔢 Two Sum - Array Problem

> **Find two numbers in an array that add up to a target value**

## 📚 Problem Statement

Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.

You may assume that each input would have **exactly one solution**, and you may not use the same element twice.

You can return the answer in any order.

## 🎯 Examples

```javascript
// Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

// Example 2:
Input: nums = [3,2,4], target = 6
Output: [1,2]

// Example 3:
Input: nums = [3,3], target = 6
Output: [0,1]
```

## 🚀 Solutions

### **Solution 1: Brute Force (O(n²))**

```javascript
/**
 * Brute Force Approach
 * Time Complexity: O(n²)
 * Space Complexity: O(1)
 */
function twoSumBruteForce(nums, target) {
  for (let i = 0; i < nums.length; i++) {
    for (let j = i + 1; j < nums.length; j++) {
      if (nums[i] + nums[j] === target) {
        return [i, j];
      }
    }
  }
  return [];
}

// Test cases
console.log(twoSumBruteForce([2, 7, 11, 15], 9)); // [0, 1]
console.log(twoSumBruteForce([3, 2, 4], 6)); // [1, 2]
console.log(twoSumBruteForce([3, 3], 6)); // [0, 1]
```

### **Solution 2: Hash Map (O(n)) - Optimal**

```javascript
/**
 * Hash Map Approach - Optimal Solution
 * Time Complexity: O(n)
 * Space Complexity: O(n)
 */
function twoSum(nums, target) {
  const numMap = new Map();

  for (let i = 0; i < nums.length; i++) {
    const complement = target - nums[i];

    if (numMap.has(complement)) {
      return [numMap.get(complement), i];
    }

    numMap.set(nums[i], i);
  }

  return [];
}

// Test cases
console.log(twoSum([2, 7, 11, 15], 9)); // [0, 1]
console.log(twoSum([3, 2, 4], 6)); // [1, 2]
console.log(twoSum([3, 3], 6)); // [0, 1]
```

### **Solution 3: Two Pointers (O(n log n)) - For Sorted Array**

```javascript
/**
 * Two Pointers Approach - For Sorted Array
 * Time Complexity: O(n log n) due to sorting
 * Space Complexity: O(1)
 */
function twoSumTwoPointers(nums, target) {
  // Create array with original indices
  const indexedNums = nums.map((num, index) => ({ value: num, index }));

  // Sort by value
  indexedNums.sort((a, b) => a.value - b.value);

  let left = 0;
  let right = indexedNums.length - 1;

  while (left < right) {
    const sum = indexedNums[left].value + indexedNums[right].value;

    if (sum === target) {
      return [indexedNums[left].index, indexedNums[right].index];
    } else if (sum < target) {
      left++;
    } else {
      right--;
    }
  }

  return [];
}

// Test cases
console.log(twoSumTwoPointers([2, 7, 11, 15], 9)); // [0, 1]
console.log(twoSumTwoPointers([3, 2, 4], 6)); // [1, 2]
console.log(twoSumTwoPointers([3, 3], 6)); // [0, 1]
```

## 🧪 Test Cases

```javascript
// Comprehensive test suite
function runTests() {
  const testCases = [
    {
      input: { nums: [2, 7, 11, 15], target: 9 },
      expected: [0, 1],
      description: "Basic case",
    },
    {
      input: { nums: [3, 2, 4], target: 6 },
      expected: [1, 2],
      description: "Different indices",
    },
    {
      input: { nums: [3, 3], target: 6 },
      expected: [0, 1],
      description: "Same numbers",
    },
    {
      input: { nums: [1, 2, 3, 4, 5], target: 8 },
      expected: [2, 4],
      description: "Larger array",
    },
    {
      input: { nums: [-1, -2, -3, -4, -5], target: -8 },
      expected: [2, 4],
      description: "Negative numbers",
    },
    {
      input: { nums: [0, 4, 3, 0], target: 0 },
      expected: [0, 3],
      description: "Zero values",
    },
  ];

  testCases.forEach((testCase, index) => {
    const result = twoSum(testCase.input.nums, testCase.input.target);
    const passed =
      JSON.stringify(result.sort()) ===
      JSON.stringify(testCase.expected.sort());

    console.log(`Test ${index + 1}: ${testCase.description}`);
    console.log(
      `Input: nums = [${testCase.input.nums}], target = ${testCase.input.target}`
    );
    console.log(`Expected: [${testCase.expected}], Got: [${result}]`);
    console.log(`Result: ${passed ? "✅ PASS" : "❌ FAIL"}\n`);
  });
}

runTests();
```

## 📊 Performance Analysis

```javascript
// Performance comparison
function performanceTest() {
  const largeArray = Array.from({ length: 10000 }, (_, i) => i);
  const target = 19999; // Last two elements

  console.log("Performance Test with 10,000 elements:");

  // Brute Force
  console.time("Brute Force");
  twoSumBruteForce(largeArray, target);
  console.timeEnd("Brute Force");

  // Hash Map
  console.time("Hash Map");
  twoSum(largeArray, target);
  console.timeEnd("Hash Map");

  // Two Pointers
  console.time("Two Pointers");
  twoSumTwoPointers(largeArray, target);
  console.timeEnd("Two Pointers");
}

performanceTest();
```

## 🎯 Variations

### **Variation 1: Two Sum II - Input Array is Sorted**

```javascript
/**
 * Two Sum II - Input Array is Sorted
 * Given a sorted array, find two numbers that add up to target
 * Return 1-indexed positions
 */
function twoSumSorted(nums, target) {
  let left = 0;
  let right = nums.length - 1;

  while (left < right) {
    const sum = nums[left] + nums[right];

    if (sum === target) {
      return [left + 1, right + 1]; // 1-indexed
    } else if (sum < target) {
      left++;
    } else {
      right--;
    }
  }

  return [];
}

// Test
console.log(twoSumSorted([2, 7, 11, 15], 9)); // [1, 2]
```

### **Variation 2: Two Sum - All Pairs**

```javascript
/**
 * Find all pairs that add up to target
 * Return all possible combinations
 */
function twoSumAllPairs(nums, target) {
  const result = [];
  const seen = new Set();

  for (let i = 0; i < nums.length; i++) {
    const complement = target - nums[i];

    if (seen.has(complement)) {
      // Find the index of complement
      const complementIndex = nums.indexOf(complement);
      result.push([complementIndex, i]);
    }

    seen.add(nums[i]);
  }

  return result;
}

// Test
console.log(twoSumAllPairs([1, 2, 3, 4, 5], 6)); // [[0, 4], [1, 3]]
```

### **Variation 3: Two Sum - Closest to Target**

```javascript
/**
 * Find two numbers whose sum is closest to target
 * Return the sum, not the indices
 */
function twoSumClosest(nums, target) {
  nums.sort((a, b) => a - b);

  let left = 0;
  let right = nums.length - 1;
  let closestSum = nums[0] + nums[1];
  let minDiff = Math.abs(closestSum - target);

  while (left < right) {
    const sum = nums[left] + nums[right];
    const diff = Math.abs(sum - target);

    if (diff < minDiff) {
      minDiff = diff;
      closestSum = sum;
    }

    if (sum < target) {
      left++;
    } else {
      right--;
    }
  }

  return closestSum;
}

// Test
console.log(twoSumClosest([-1, 2, 1, -4], 1)); // 2
```

## 🎯 Interview Tips

### **Key Points to Mention:**

1. **Hash Map is optimal** for unsorted arrays (O(n) time, O(n) space)
2. **Two Pointers** work well for sorted arrays (O(n) time, O(1) space)
3. **Brute Force** is simple but inefficient (O(n²) time)
4. **Edge cases**: Empty array, no solution, duplicate numbers

### **Follow-up Questions:**

1. What if the array is sorted?
2. What if we need to find all pairs?
3. What if we need the closest sum to target?
4. How would you handle very large arrays?

### **Common Mistakes:**

1. Using the same element twice
2. Not handling edge cases
3. Returning values instead of indices
4. Not considering space complexity

---

**🎉 Master the Two Sum problem to build a strong foundation for array manipulation!**

**Good luck with your coding interviews! 🚀**
