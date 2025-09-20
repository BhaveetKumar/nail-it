# 📊 **Arrays: Complete Guide with Node.js**

> **Master array algorithms and data structures for technical interviews**

## 🎯 **Learning Objectives**

- Master array manipulation and algorithms
- Understand time and space complexity
- Solve 80+ array problems with optimal solutions
- Build problem-solving intuition
- Prepare for FAANG interviews

## 📚 **Topics Covered**

### **🔢 Basic Operations**
- [**Two Sum**](TwoSum.md) - Hash map approach
- [**Best Time to Buy and Sell Stock**](BestTimeToBuySellStock.md) - Greedy algorithm
- [**Contains Duplicate**](ContainsDuplicate.md) - Set operations
- [**Product of Array Except Self**](ProductOfArrayExceptSelf.md) - Prefix/suffix products

### **🔄 Two Pointers**
- [**Two Sum II**](TwoSumII.md) - Sorted array two pointers
- [**3Sum**](ThreeSum.md) - Three pointer technique
- [**Container With Most Water**](ContainerWithMostWater.md) - Area maximization
- [**Remove Duplicates**](../../../algorithms/Arrays/RemoveDuplicates.md) - In-place deduplication

### **🪟 Sliding Window**
- [**Maximum Subarray**](MaximumSubarray.md) - Kadane's algorithm
- [**Sliding Window Maximum**](../../../algorithms/SlidingWindow/SlidingWindowMaximum.md) - Deque technique
- [**Longest Substring Without Repeating Characters**](../../../algorithms/Arrays/LongestSubstring.md) - Character frequency
- [**Minimum Window Substring**](../../../algorithms/SlidingWindow/MinimumWindowSubstring.md) - Template matching

### **📈 Prefix Sum**
- [**Range Sum Query**](../../../algorithms/Arrays/RangeSumQuery.md) - Immutable array
- [**Subarray Sum Equals K**](../../../algorithms/Arrays/SubarraySumEqualsK.md) - Hash map + prefix sum
- [**Continuous Subarray Sum**](ContinuousSubarraySum.md) - Modulo arithmetic
- [**Maximum Size Subarray Sum Equals K**](MaxSizeSubarraySumK.md) - Size optimization

### **🗺️ Hash Maps**
- [**Group Anagrams**](../../../algorithms/Strings/GroupAnagrams.md) - String sorting
- [**Top K Frequent Elements**](../../../algorithms/Heap/TopKFrequentElements.md) - Frequency counting
- [**Intersection of Two Arrays**](IntersectionOfTwoArrays.md) - Set operations
- [**Valid Anagram**](../../../algorithms/Strings/ValidAnagram.md) - Character frequency

### **🔄 Rotation & Transformation**
- [**Rotate Array**](RotateArray.md) - In-place rotation
- [**Search in Rotated Sorted Array**](SearchRotatedSortedArray.md) - Binary search
- [**Find Minimum in Rotated Sorted Array**](FindMinRotatedSortedArray.md) - Pivot finding
- [**Spiral Matrix**](../../../algorithms/Arrays/SpiralMatrix.md) - Matrix traversal

### **🎯 Advanced Problems**
- [**Merge Intervals**](MergeIntervals.md) - Interval merging
- [**Insert Interval**](InsertInterval.md) - Interval insertion
- [**Meeting Rooms**](../../../algorithms/Greedy/MeetingRooms.md) - Scheduling problems
- [**Trapping Rain Water**](../../../algorithms/Arrays/TrappingRainWater.md) - Water trapping

## 🚀 **Getting Started**

### **Prerequisites**
- JavaScript ES6+ fundamentals
- Basic understanding of time complexity
- Familiarity with array methods
- Problem-solving mindset

### **Learning Path**

#### **Week 1: Basic Operations**
1. Start with simple array problems
2. Learn hash map techniques
3. Practice two pointer approach
4. Master prefix sum concepts

#### **Week 2: Advanced Techniques**
1. Study sliding window patterns
2. Learn rotation algorithms
3. Practice interval problems
4. Master matrix operations

## 📊 **Problem Statistics**

- **Total Problems**: 80+
- **Easy**: 30+ problems
- **Medium**: 35+ problems
- **Hard**: 15+ problems
- **Patterns**: 8 major patterns

## 🎯 **Common Patterns**

### **1. Two Pointers**
```javascript
// Template for two pointers
function twoPointers(arr) {
    let left = 0;
    let right = arr.length - 1;
    
    while (left < right) {
        // Process current pair
        if (condition) {
            left++;
        } else {
            right--;
        }
    }
    
    return result;
}
```

### **2. Sliding Window**
```javascript
// Template for sliding window
function slidingWindow(arr, k) {
    let windowSum = 0;
    let maxSum = 0;
    
    // Initialize first window
    for (let i = 0; i < k; i++) {
        windowSum += arr[i];
    }
    
    // Slide the window
    for (let i = k; i < arr.length; i++) {
        windowSum = windowSum - arr[i - k] + arr[i];
        maxSum = Math.max(maxSum, windowSum);
    }
    
    return maxSum;
}
```

### **3. Hash Map**
```javascript
// Template for hash map
function hashMapApproach(arr) {
    const map = new Map();
    
    for (let i = 0; i < arr.length; i++) {
        if (map.has(target - arr[i])) {
            return [map.get(target - arr[i]), i];
        }
        map.set(arr[i], i);
    }
    
    return [];
}
```

## 🔧 **Code Standards**

### **Problem Template**
```javascript
/**
 * Problem: [Problem Name]
 * Difficulty: [Easy/Medium/Hard]
 * Time Complexity: O(n)
 * Space Complexity: O(1)
 * 
 * @param {number[]} nums
 * @return {number}
 */
function solution(nums) {
    // Implementation here
    return result;
}

// Test cases
console.log(solution([1, 2, 3])); // Expected output
console.log(solution([4, 5, 6])); // Expected output
```

### **Testing Framework**
```javascript
// Test helper function
function test(actual, expected, testName) {
    if (JSON.stringify(actual) === JSON.stringify(expected)) {
        console.log(`✅ ${testName} passed`);
    } else {
        console.log(`❌ ${testName} failed`);
        console.log(`Expected: ${expected}`);
        console.log(`Actual: ${actual}`);
    }
}

// Usage
test(solution([1, 2, 3]), [1, 2, 3], "Basic test");
```

## 📈 **Progress Tracking**

### **Beginner Level (0-20 problems)**
- [ ] Two Sum
- [ ] Best Time to Buy and Sell Stock
- [ ] Contains Duplicate
- [ ] Valid Anagram
- [ ] Group Anagrams

### **Intermediate Level (20-50 problems)**
- [ ] 3Sum
- [ ] Container With Most Water
- [ ] Maximum Subarray
- [ ] Product of Array Except Self
- [ ] Spiral Matrix

### **Advanced Level (50+ problems)**
- [ ] Trapping Rain Water
- [ ] Merge Intervals
- [ ] Sliding Window Maximum
- [ ] Minimum Window Substring
- [ ] Find Minimum in Rotated Sorted Array

## 🎯 **Interview Tips**

### **Google Interview Focus**
- **Optimal Solutions**: Always aim for the best time/space complexity
- **Edge Cases**: Handle empty arrays, single elements, duplicates
- **Code Quality**: Write clean, readable, maintainable code
- **Explanation**: Clearly explain your approach and reasoning

### **Meta Interview Focus**
- **Problem Solving**: Break down complex problems step by step
- **Communication**: Explain your thought process clearly
- **Optimization**: Discuss trade-offs between different approaches
- **Testing**: Consider various test cases and edge scenarios

### **Amazon Interview Focus**
- **System Design**: Think about real-world applications
- **Performance**: Focus on efficiency and scalability
- **Reliability**: Handle error cases and edge conditions
- **Maintainability**: Write production-ready code

## 📚 **Resources**

### **Practice Platforms**
- LeetCode
- HackerRank
- CodeSignal
- Pramp

### **Study Materials**
- "Cracking the Coding Interview"
- "Elements of Programming Interviews"
- LeetCode discussion forums
- YouTube algorithm channels

---

**🚀 Ready to master array algorithms and ace your technical interviews!**
