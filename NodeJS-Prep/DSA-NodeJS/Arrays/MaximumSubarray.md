# 📈 Maximum Subarray (Kadane's Algorithm)

> **Classic dynamic programming problem for finding maximum sum subarray**

## 📋 **Problem Statement**

Given an integer array `nums`, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

A **subarray** is a contiguous part of an array.

## 🎯 **Examples**

```javascript
// Example 1
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.

// Example 2
Input: nums = [1]
Output: 1
Explanation: The subarray [1] has the largest sum = 1.

// Example 3
Input: nums = [5,4,-1,7,8]
Output: 23
Explanation: [5,4,-1,7,8] has the largest sum = 23.
```

## 🧠 **Approach**

### **Brute Force Approach**
- Check all possible subarrays
- Time Complexity: O(n³)
- Space Complexity: O(1)

### **Kadane's Algorithm (Optimal)**
- Keep track of current sum and maximum sum
- Reset current sum when it becomes negative
- Time Complexity: O(n)
- Space Complexity: O(1)

## 🔍 **Dry Run**

```
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]

Step 1: currentSum = -2, maxSum = -2
Step 2: currentSum = max(-2 + 1, 1) = 1, maxSum = 1
Step 3: currentSum = max(1 + (-3), -3) = -2, maxSum = 1
Step 4: currentSum = max(-2 + 4, 4) = 4, maxSum = 4
Step 5: currentSum = max(4 + (-1), -1) = 3, maxSum = 4
Step 6: currentSum = max(3 + 2, 2) = 5, maxSum = 5
Step 7: currentSum = max(5 + 1, 1) = 6, maxSum = 6
Step 8: currentSum = max(6 + (-5), -5) = 1, maxSum = 6
Step 9: currentSum = max(1 + 4, 4) = 5, maxSum = 6

Result: 6
```

## 💻 **Solution**

### **Kadane's Algorithm (Optimal)**

```javascript
/**
 * Maximum Subarray - Kadane's Algorithm
 * Time Complexity: O(n)
 * Space Complexity: O(1)
 * 
 * @param {number[]} nums
 * @return {number}
 */
function maxSubArray(nums) {
    let currentSum = nums[0];
    let maxSum = nums[0];
    
    for (let i = 1; i < nums.length; i++) {
        // Either extend the existing subarray or start a new one
        currentSum = Math.max(nums[i], currentSum + nums[i]);
        maxSum = Math.max(maxSum, currentSum);
    }
    
    return maxSum;
}

// Alternative implementation with reset logic
function maxSubArrayReset(nums) {
    let currentSum = 0;
    let maxSum = nums[0];
    
    for (let i = 0; i < nums.length; i++) {
        currentSum += nums[i];
        maxSum = Math.max(maxSum, currentSum);
        
        // Reset if current sum becomes negative
        if (currentSum < 0) {
            currentSum = 0;
        }
    }
    
    return maxSum;
}
```

### **Brute Force Solution**

```javascript
/**
 * Maximum Subarray - Brute Force
 * Time Complexity: O(n²)
 * Space Complexity: O(1)
 * 
 * @param {number[]} nums
 * @return {number}
 */
function maxSubArrayBruteForce(nums) {
    let maxSum = nums[0];
    
    for (let i = 0; i < nums.length; i++) {
        let currentSum = 0;
        for (let j = i; j < nums.length; j++) {
            currentSum += nums[j];
            maxSum = Math.max(maxSum, currentSum);
        }
    }
    
    return maxSum;
}
```

### **Divide and Conquer Solution**

```javascript
/**
 * Maximum Subarray - Divide and Conquer
 * Time Complexity: O(n log n)
 * Space Complexity: O(log n)
 * 
 * @param {number[]} nums
 * @return {number}
 */
function maxSubArrayDivideConquer(nums) {
    return divideConquer(nums, 0, nums.length - 1);
}

function divideConquer(nums, left, right) {
    if (left === right) {
        return nums[left];
    }
    
    const mid = Math.floor((left + right) / 2);
    
    // Find max subarray in left half
    const leftMax = divideConquer(nums, left, mid);
    
    // Find max subarray in right half
    const rightMax = divideConquer(nums, mid + 1, right);
    
    // Find max subarray crossing the middle
    const crossMax = maxCrossingSubarray(nums, left, mid, right);
    
    return Math.max(leftMax, rightMax, crossMax);
}

function maxCrossingSubarray(nums, left, mid, right) {
    let leftSum = -Infinity;
    let sum = 0;
    
    // Find max sum from mid to left
    for (let i = mid; i >= left; i--) {
        sum += nums[i];
        leftSum = Math.max(leftSum, sum);
    }
    
    let rightSum = -Infinity;
    sum = 0;
    
    // Find max sum from mid+1 to right
    for (let i = mid + 1; i <= right; i++) {
        sum += nums[i];
        rightSum = Math.max(rightSum, sum);
    }
    
    return leftSum + rightSum;
}
```

## 🧪 **Test Cases**

```javascript
// Test helper function
function test(actual, expected, testName) {
    const isEqual = actual === expected;
    console.log(`${isEqual ? '✅' : '❌'} ${testName}`);
    if (!isEqual) {
        console.log(`Expected: ${expected}`);
        console.log(`Actual: ${actual}`);
    }
}

// Test cases
test(maxSubArray([-2,1,-3,4,-1,2,1,-5,4]), 6, "Example 1");
test(maxSubArray([1]), 1, "Example 2");
test(maxSubArray([5,4,-1,7,8]), 23, "Example 3");
test(maxSubArray([-1]), -1, "Single negative");
test(maxSubArray([-2,-1]), -1, "All negative");
test(maxSubArray([1,2,3,4,5]), 15, "All positive");
test(maxSubArray([1,-1,1,-1,1]), 1, "Alternating");
```

## 📊 **Complexity Analysis**

### **Kadane's Algorithm**
- **Time Complexity**: O(n) - Single pass through array
- **Space Complexity**: O(1) - Only using constant extra space
- **Best Case**: O(n) - Always need to check all elements
- **Worst Case**: O(n) - Same as best case

### **Brute Force**
- **Time Complexity**: O(n²) - Nested loops
- **Space Complexity**: O(1) - No extra space
- **Best Case**: O(n²) - Always check all subarrays
- **Worst Case**: O(n²) - Same as best case

### **Divide and Conquer**
- **Time Complexity**: O(n log n) - Recursive division
- **Space Complexity**: O(log n) - Recursion stack
- **Best Case**: O(n log n) - Always divide
- **Worst Case**: O(n log n) - Same as best case

## 🎯 **Key Insights**

1. **Greedy Choice**: Start new subarray when current sum becomes negative
2. **Optimal Substructure**: Maximum subarray ending at position i
3. **Reset Strategy**: Don't carry negative sums forward
4. **Single Pass**: Can solve in one iteration
5. **Dynamic Programming**: Current sum depends on previous sum

## 🔄 **Variations**

### **Maximum Subarray with Indices**
```javascript
// Return the actual subarray indices
function maxSubArrayWithIndices(nums) {
    let currentSum = nums[0];
    let maxSum = nums[0];
    let start = 0, end = 0, tempStart = 0;
    
    for (let i = 1; i < nums.length; i++) {
        if (currentSum < 0) {
            currentSum = nums[i];
            tempStart = i;
        } else {
            currentSum += nums[i];
        }
        
        if (currentSum > maxSum) {
            maxSum = currentSum;
            start = tempStart;
            end = i;
        }
    }
    
    return { sum: maxSum, start, end, subarray: nums.slice(start, end + 1) };
}
```

### **Maximum Subarray Product**
```javascript
// Maximum product subarray
function maxProduct(nums) {
    let maxSoFar = nums[0];
    let maxEndingHere = nums[0];
    let minEndingHere = nums[0];
    
    for (let i = 1; i < nums.length; i++) {
        const temp = maxEndingHere;
        maxEndingHere = Math.max(nums[i], Math.max(maxEndingHere * nums[i], minEndingHere * nums[i]));
        minEndingHere = Math.min(nums[i], Math.min(temp * nums[i], minEndingHere * nums[i]));
        maxSoFar = Math.max(maxSoFar, maxEndingHere);
    }
    
    return maxSoFar;
}
```

### **Circular Maximum Subarray**
```javascript
// Maximum subarray in circular array
function maxSubarraySumCircular(nums) {
    // Case 1: Maximum subarray is not circular
    const maxNonCircular = maxSubArray(nums);
    
    // Case 2: Maximum subarray is circular
    // This means we need to find minimum subarray and subtract from total
    let totalSum = 0;
    let minSum = nums[0];
    let currentMin = nums[0];
    
    for (let i = 0; i < nums.length; i++) {
        totalSum += nums[i];
        currentMin = Math.min(nums[i], currentMin + nums[i]);
        minSum = Math.min(minSum, currentMin);
    }
    
    const maxCircular = totalSum - minSum;
    
    // Handle edge case where all numbers are negative
    return maxCircular === 0 ? maxNonCircular : Math.max(maxNonCircular, maxCircular);
}
```

## 🎓 **Interview Tips**

### **Google Interview**
- **Start with brute force**: Show understanding of the problem
- **Optimize step by step**: Explain Kadane's algorithm intuition
- **Handle edge cases**: All negative numbers, single element
- **Code quality**: Clean implementation with clear variable names

### **Meta Interview**
- **Think out loud**: Explain the greedy approach
- **Visualize**: Draw the array and trace through the algorithm
- **Test thoroughly**: Walk through examples step by step
- **Consider variations**: What if we need the actual subarray?

### **Amazon Interview**
- **Real-world context**: How would this apply to stock prices?
- **Optimization**: Can we solve it in O(1) space?
- **Edge cases**: What if the array is empty or has one element?
- **Production code**: Write robust, well-tested code

## 📚 **Related Problems**

- [**Maximum Product Subarray**](./MaximumProductSubarray.md) - Product instead of sum
- [**Circular Array Loop**](./CircularArrayLoop.md) - Circular array problems
- [**Best Time to Buy and Sell Stock**](./BestTimeToBuySellStock.md) - Similar DP pattern
- [**House Robber**](./HouseRobber.md) - Similar optimization problem

## 🎉 **Summary**

Maximum Subarray teaches:
- **Dynamic programming** fundamentals
- **Greedy algorithms** and local optimization
- **Kadane's algorithm** for optimal solution
- **Space optimization** techniques

This problem is fundamental for understanding dynamic programming and appears in many variations!

---

**🚀 Ready to solve more array problems? Check out the next problem!**
