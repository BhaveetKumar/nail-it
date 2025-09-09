# 🔢 Two Sum

> **Classic array problem using hash map for optimal solution**

## 📋 **Problem Statement**

Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.

You may assume that each input would have **exactly one solution**, and you may not use the same element twice.

You can return the answer in any order.

## 🎯 **Examples**

```javascript
// Example 1
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

// Example 2
Input: nums = [3,2,4], target = 6
Output: [1,2]
Explanation: Because nums[2] + nums[1] == 6, we return [1, 2].

// Example 3
Input: nums = [3,3], target = 6
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 6, we return [0, 1].
```

## 🧠 **Approach**

### **Brute Force Approach**
- Check every pair of numbers in the array
- Time Complexity: O(n²)
- Space Complexity: O(1)

### **Hash Map Approach (Optimal)**
- Use a hash map to store numbers and their indices
- For each number, check if complement exists in map
- Time Complexity: O(n)
- Space Complexity: O(n)

## 🔍 **Dry Run**

```
nums = [2, 7, 11, 15], target = 9

Step 1: i=0, nums[0]=2, complement=9-2=7
        map = {}, complement not found
        map = {2: 0}

Step 2: i=1, nums[1]=7, complement=9-7=2
        map = {2: 0}, complement found!
        return [0, 1]
```

## 💻 **Solution**

### **Hash Map Solution (Optimal)**

```javascript
/**
 * Two Sum - Hash Map Approach
 * Time Complexity: O(n)
 * Space Complexity: O(n)
 * 
 * @param {number[]} nums
 * @param {number} target
 * @return {number[]}
 */
function twoSum(nums, target) {
    const map = new Map();
    
    for (let i = 0; i < nums.length; i++) {
        const complement = target - nums[i];
        
        if (map.has(complement)) {
            return [map.get(complement), i];
        }
        
        map.set(nums[i], i);
    }
    
    return [];
}

// Alternative implementation using object
function twoSumObject(nums, target) {
    const numMap = {};
    
    for (let i = 0; i < nums.length; i++) {
        const complement = target - nums[i];
        
        if (complement in numMap) {
            return [numMap[complement], i];
        }
        
        numMap[nums[i]] = i;
    }
    
    return [];
}
```

### **Brute Force Solution**

```javascript
/**
 * Two Sum - Brute Force Approach
 * Time Complexity: O(n²)
 * Space Complexity: O(1)
 * 
 * @param {number[]} nums
 * @param {number} target
 * @return {number[]}
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
```

## 🧪 **Test Cases**

```javascript
// Test helper function
function test(actual, expected, testName) {
    const isEqual = JSON.stringify(actual.sort()) === JSON.stringify(expected.sort());
    console.log(`${isEqual ? '✅' : '❌'} ${testName}`);
    if (!isEqual) {
        console.log(`Expected: ${expected}`);
        console.log(`Actual: ${actual}`);
    }
}

// Test cases
test(twoSum([2, 7, 11, 15], 9), [0, 1], "Example 1");
test(twoSum([3, 2, 4], 6), [1, 2], "Example 2");
test(twoSum([3, 3], 6), [0, 1], "Example 3");
test(twoSum([1, 2, 3, 4, 5], 8), [2, 4], "Custom test 1");
test(twoSum([-1, -2, -3, -4, -5], -8), [2, 4], "Negative numbers");
test(twoSum([0, 4, 3, 0], 0), [0, 3], "Zero target");
test(twoSum([1, 2], 4), [], "No solution");
```

## 📊 **Complexity Analysis**

### **Hash Map Approach**
- **Time Complexity**: O(n) - Single pass through array
- **Space Complexity**: O(n) - Hash map stores up to n elements
- **Best Case**: O(1) - Solution found at index 1
- **Worst Case**: O(n) - Solution found at last index

### **Brute Force Approach**
- **Time Complexity**: O(n²) - Nested loops
- **Space Complexity**: O(1) - No extra space used
- **Best Case**: O(1) - Solution found at first pair
- **Worst Case**: O(n²) - Solution found at last pair

## 🎯 **Key Insights**

1. **Hash Map Trade-off**: Use extra space to reduce time complexity
2. **One Pass**: Can find solution in single iteration
3. **Complement Strategy**: Look for target - current number
4. **Index Storage**: Store both value and index in map
5. **Early Return**: Return immediately when solution found

## 🔄 **Variations**

### **Two Sum II - Sorted Array**
```javascript
// Array is sorted, use two pointers
function twoSumSorted(numbers, target) {
    let left = 0;
    let right = numbers.length - 1;
    
    while (left < right) {
        const sum = numbers[left] + numbers[right];
        
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
```

### **Two Sum - Return Values**
```javascript
// Return the actual values instead of indices
function twoSumValues(nums, target) {
    const map = new Map();
    
    for (let i = 0; i < nums.length; i++) {
        const complement = target - nums[i];
        
        if (map.has(complement)) {
            return [complement, nums[i]];
        }
        
        map.set(nums[i], i);
    }
    
    return [];
}
```

### **Two Sum - All Pairs**
```javascript
// Return all possible pairs
function twoSumAllPairs(nums, target) {
    const map = new Map();
    const result = [];
    
    for (let i = 0; i < nums.length; i++) {
        const complement = target - nums[i];
        
        if (map.has(complement)) {
            result.push([map.get(complement), i]);
        }
        
        map.set(nums[i], i);
    }
    
    return result;
}
```

## 🎓 **Interview Tips**

### **Google Interview**
- **Start with brute force**: Show you understand the problem
- **Optimize step by step**: Explain the trade-off between time and space
- **Handle edge cases**: Empty array, no solution, duplicate numbers
- **Code quality**: Write clean, readable code with proper variable names

### **Meta Interview**
- **Think out loud**: Explain your thought process
- **Consider alternatives**: Discuss different approaches
- **Test your solution**: Walk through examples
- **Optimize further**: Can you improve space or time complexity?

### **Amazon Interview**
- **Real-world context**: How would this apply to a payment system?
- **Scalability**: What if the array has millions of elements?
- **Error handling**: What if input is invalid?
- **Production code**: Write code you'd be comfortable deploying

## 📚 **Related Problems**

- [**Two Sum II**](./TwoSumII.md) - Sorted array version
- [**3Sum**](./ThreeSum.md) - Three numbers that sum to target
- [**4Sum**](./FourSum.md) - Four numbers that sum to target
- [**Two Sum - Data Structure**](./TwoSumDataStructure.md) - Design a data structure

## 🎉 **Summary**

Two Sum is a fundamental problem that teaches:
- **Hash map usage** for O(1) lookups
- **Space-time trade-offs** in algorithm design
- **One-pass solutions** for efficiency
- **Complement strategy** for sum problems

Master this problem as it forms the foundation for many other array and sum-related problems!

---

**🚀 Ready to solve more array problems? Check out the next problem!**
