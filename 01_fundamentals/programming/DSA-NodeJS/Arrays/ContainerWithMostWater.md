# 🏺 Container With Most Water

> **Classic two pointers problem for finding maximum area**

## 📋 **Problem Statement**

Given `n` non-negative integers `a1, a2, ..., an`, where each represents a point at coordinate `(i, ai)`. `n` vertical lines are drawn such that the two endpoints of the line `i` is at `(i, 0)` and `(i, ai)`.

Find two lines, which, together with the x-axis forms a container, such that the container contains the most water.

Notice that you may not slant the container.

## 🎯 **Examples**

```javascript
// Example 1
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. 
In this case, the max area of water (blue section) the container can contain is 49.

// Example 2
Input: height = [1,1]
Output: 1

// Example 3
Input: height = [4,3,2,1,4]
Output: 16

// Example 4
Input: height = [1,2,1]
Output: 2
```

## 🧠 **Approach**

### **Brute Force Approach**
- Check all possible pairs of lines
- Calculate area for each pair
- Time Complexity: O(n²)
- Space Complexity: O(1)

### **Two Pointers Approach (Optimal)**
- Start with two pointers at both ends
- Move the pointer with smaller height
- Keep track of maximum area
- Time Complexity: O(n)
- Space Complexity: O(1)

## 🔍 **Dry Run**

```
height = [1, 8, 6, 2, 5, 4, 8, 3, 7]

left = 0, right = 8
height[left] = 1, height[right] = 7
width = 8 - 0 = 8
area = min(1, 7) * 8 = 1 * 8 = 8
maxArea = 8

Move left pointer (height[left] < height[right])
left = 1, right = 8
height[left] = 8, height[right] = 7
width = 8 - 1 = 7
area = min(8, 7) * 7 = 7 * 7 = 49
maxArea = 49

Move right pointer (height[right] < height[left])
left = 1, right = 7
height[left] = 8, height[right] = 3
width = 7 - 1 = 6
area = min(8, 3) * 6 = 3 * 6 = 18
maxArea = 49

Move right pointer (height[right] < height[left])
left = 1, right = 6
height[left] = 8, height[right] = 8
width = 6 - 1 = 5
area = min(8, 8) * 5 = 8 * 5 = 40
maxArea = 49

Move left pointer (height[left] == height[right])
left = 2, right = 6
height[left] = 6, height[right] = 8
width = 6 - 2 = 4
area = min(6, 8) * 4 = 6 * 4 = 24
maxArea = 49

Continue until left >= right...

Result: 49
```

## 💻 **Solution**

### **Two Pointers Solution (Optimal)**

```javascript
/**
 * Container With Most Water - Two Pointers Approach
 * Time Complexity: O(n)
 * Space Complexity: O(1)
 * 
 * @param {number[]} height
 * @return {number}
 */
function maxArea(height) {
    let left = 0;
    let right = height.length - 1;
    let maxArea = 0;
    
    while (left < right) {
        // Calculate current area
        const width = right - left;
        const currentHeight = Math.min(height[left], height[right]);
        const currentArea = width * currentHeight;
        
        // Update maximum area
        maxArea = Math.max(maxArea, currentArea);
        
        // Move pointer with smaller height
        if (height[left] < height[right]) {
            left++;
        } else {
            right--;
        }
    }
    
    return maxArea;
}

// Alternative implementation with more explicit logic
function maxAreaAlternative(height) {
    let left = 0;
    let right = height.length - 1;
    let maxArea = 0;
    
    while (left < right) {
        const width = right - left;
        const leftHeight = height[left];
        const rightHeight = height[right];
        const minHeight = Math.min(leftHeight, rightHeight);
        const area = width * minHeight;
        
        maxArea = Math.max(maxArea, area);
        
        // Move the pointer with smaller height
        if (leftHeight < rightHeight) {
            left++;
        } else if (leftHeight > rightHeight) {
            right--;
        } else {
            // Both heights are equal, move both pointers
            left++;
            right--;
        }
    }
    
    return maxArea;
}
```

### **Brute Force Solution**

```javascript
/**
 * Container With Most Water - Brute Force Approach
 * Time Complexity: O(n²)
 * Space Complexity: O(1)
 * 
 * @param {number[]} height
 * @return {number}
 */
function maxAreaBruteForce(height) {
    let maxArea = 0;
    
    for (let i = 0; i < height.length; i++) {
        for (let j = i + 1; j < height.length; j++) {
            const width = j - i;
            const minHeight = Math.min(height[i], height[j]);
            const area = width * minHeight;
            maxArea = Math.max(maxArea, area);
        }
    }
    
    return maxArea;
}

// Optimized brute force with early termination
function maxAreaBruteForceOptimized(height) {
    let maxArea = 0;
    
    for (let i = 0; i < height.length; i++) {
        // Skip if current height is 0
        if (height[i] === 0) continue;
        
        // Calculate minimum area needed to beat current max
        const minWidth = Math.ceil(maxArea / height[i]);
        
        for (let j = i + minWidth; j < height.length; j++) {
            const width = j - i;
            const minHeight = Math.min(height[i], height[j]);
            const area = width * minHeight;
            maxArea = Math.max(maxArea, area);
        }
    }
    
    return maxArea;
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
test(maxArea([1,8,6,2,5,4,8,3,7]), 49, "Example 1");
test(maxArea([1,1]), 1, "Example 2");
test(maxArea([4,3,2,1,4]), 16, "Example 3");
test(maxArea([1,2,1]), 2, "Example 4");
test(maxArea([1,2,4,3]), 4, "Custom test 1");
test(maxArea([2,3,4,5,18,17,6]), 17, "Custom test 2");
test(maxArea([1,1,1,1,1]), 4, "All same height");
test(maxArea([1,2,3,4,5]), 6, "Increasing heights");
test(maxArea([5,4,3,2,1]), 6, "Decreasing heights");
test(maxArea([1]), 0, "Single element");
test(maxArea([]), 0, "Empty array");
```

## 📊 **Complexity Analysis**

### **Two Pointers Approach**
- **Time Complexity**: O(n) - Single pass through array
- **Space Complexity**: O(1) - Only using constant extra space
- **Best Case**: O(n) - Always need to check all elements
- **Worst Case**: O(n) - Same as best case

### **Brute Force Approach**
- **Time Complexity**: O(n²) - Nested loops
- **Space Complexity**: O(1) - Only using constant extra space
- **Best Case**: O(n²) - Always check all pairs
- **Worst Case**: O(n²) - Same as best case

## 🎯 **Key Insights**

1. **Two Pointers**: Start from both ends and move inward
2. **Greedy Choice**: Always move the pointer with smaller height
3. **Mathematical Insight**: Area = min(height[left], height[right]) × width
4. **Optimal Substructure**: Maximum area is found by comparing all possible containers
5. **Proof of Correctness**: Moving the smaller pointer can only increase the area

## 🔄 **Variations**

### **Trapping Rain Water**
```javascript
// Similar problem: Trapping Rain Water
function trap(height) {
    let left = 0;
    let right = height.length - 1;
    let leftMax = 0;
    let rightMax = 0;
    let water = 0;
    
    while (left < right) {
        if (height[left] < height[right]) {
            if (height[left] >= leftMax) {
                leftMax = height[left];
            } else {
                water += leftMax - height[left];
            }
            left++;
        } else {
            if (height[right] >= rightMax) {
                rightMax = height[right];
            } else {
                water += rightMax - height[right];
            }
            right--;
        }
    }
    
    return water;
}
```

### **Largest Rectangle in Histogram**
```javascript
// Related problem: Largest Rectangle in Histogram
function largestRectangleArea(heights) {
    const stack = [];
    let maxArea = 0;
    
    for (let i = 0; i <= heights.length; i++) {
        const h = i === heights.length ? 0 : heights[i];
        
        while (stack.length > 0 && h < heights[stack[stack.length - 1]]) {
            const height = heights[stack.pop()];
            const width = stack.length === 0 ? i : i - stack[stack.length - 1] - 1;
            maxArea = Math.max(maxArea, height * width);
        }
        
        stack.push(i);
    }
    
    return maxArea;
}
```

### **Maximum Product Subarray**
```javascript
// Similar pattern: Maximum Product Subarray
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

## 🎓 **Interview Tips**

### **Google Interview**
- **Start with brute force**: Show understanding of the problem
- **Optimize step by step**: Explain the two pointers approach
- **Handle edge cases**: Empty array, single element, all same heights
- **Code quality**: Clean implementation with proper variable names

### **Meta Interview**
- **Think out loud**: Explain the greedy approach
- **Visualize**: Draw the container and trace through the algorithm
- **Test thoroughly**: Walk through examples step by step
- **Consider variations**: What if we need to find the actual container?

### **Amazon Interview**
- **Real-world context**: How would this apply to water storage systems?
- **Optimization**: Can we solve it in O(1) space?
- **Edge cases**: What if heights can be negative?
- **Production code**: Write robust, well-tested code

## 📚 **Related Problems**

- [**Two Sum**](TwoSum.md/) - Two numbers that sum to target
- [**3Sum**](ThreeSum.md/) - Three numbers that sum to target
- [**Trapping Rain Water**](TrappingRainWater.md/) - Water trapping problem
- [**Largest Rectangle in Histogram**](LargestRectangleInHistogram.md/) - Rectangle area problem

## 🎉 **Summary**

Container With Most Water teaches:
- **Two pointers technique** for array problems
- **Greedy algorithms** and local optimization
- **Mathematical insight** for area calculations
- **Proof of correctness** for algorithm optimization

This problem is fundamental for understanding two pointers and appears in many variations!

---

**🚀 Ready to solve more array problems? Check out the next problem!**
