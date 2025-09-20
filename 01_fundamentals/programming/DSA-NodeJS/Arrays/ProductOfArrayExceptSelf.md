# 📊 Product of Array Except Self

> **Classic array problem using prefix and suffix products**

## 📋 **Problem Statement**

Given an integer array `nums`, return an array `answer` such that `answer[i]` is equal to the product of all the elements of `nums` except `nums[i]`.

The product of any prefix or suffix of `nums` is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operator.

## 🎯 **Examples**

```javascript
// Example 1
Input: nums = [1,2,3,4]
Output: [24,12,8,6]
Explanation: 
- answer[0] = 2*3*4 = 24
- answer[1] = 1*3*4 = 12
- answer[2] = 1*2*4 = 8
- answer[3] = 1*2*3 = 6

// Example 2
Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]
Explanation: 
- answer[0] = 1*0*(-3)*3 = 0
- answer[1] = (-1)*0*(-3)*3 = 0
- answer[2] = (-1)*1*(-3)*3 = 9
- answer[3] = (-1)*1*0*3 = 0
- answer[4] = (-1)*1*0*(-3) = 0
```

## 🧠 **Approach**

### **Brute Force Approach**
- For each element, calculate product of all other elements
- Time Complexity: O(n²)
- Space Complexity: O(1)

### **Division Approach (Not Allowed)**
- Calculate total product, then divide by each element
- Time Complexity: O(n)
- Space Complexity: O(1)
- **Problem**: Division by zero, precision issues

### **Prefix and Suffix Products (Optimal)**
- Calculate prefix products from left to right
- Calculate suffix products from right to left
- Result[i] = prefix[i-1] * suffix[i+1]
- Time Complexity: O(n)
- Space Complexity: O(1) - excluding output array

## 🔍 **Dry Run**

```
nums = [1, 2, 3, 4]

Step 1: Calculate prefix products
prefix = [1, 1, 2, 6]
- prefix[0] = 1 (no elements before)
- prefix[1] = 1 (product of elements before index 1)
- prefix[2] = 1*2 = 2
- prefix[3] = 1*2*3 = 6

Step 2: Calculate suffix products
suffix = [24, 12, 4, 1]
- suffix[0] = 2*3*4 = 24
- suffix[1] = 3*4 = 12
- suffix[2] = 4 (product of elements after index 2)
- suffix[3] = 1 (no elements after)

Step 3: Calculate result
result[0] = prefix[0] * suffix[1] = 1 * 12 = 12
result[1] = prefix[1] * suffix[2] = 1 * 4 = 4
result[2] = prefix[2] * suffix[3] = 2 * 1 = 2
result[3] = prefix[3] * suffix[4] = 6 * 1 = 6

Wait, let me recalculate:
result[0] = 1 * (2*3*4) = 24
result[1] = 1 * (3*4) = 12
result[2] = (1*2) * 4 = 8
result[3] = (1*2*3) * 1 = 6

Result: [24, 12, 8, 6]
```

## 💻 **Solution**

### **Prefix and Suffix Products (Optimal)**

```javascript
/**
 * Product of Array Except Self - Prefix and Suffix Products
 * Time Complexity: O(n)
 * Space Complexity: O(1) - excluding output array
 * 
 * @param {number[]} nums
 * @return {number[]}
 */
function productExceptSelf(nums) {
    const n = nums.length;
    const result = new Array(n);
    
    // Calculate prefix products
    result[0] = 1;
    for (let i = 1; i < n; i++) {
        result[i] = result[i - 1] * nums[i - 1];
    }
    
    // Calculate suffix products and multiply with prefix
    let suffix = 1;
    for (let i = n - 1; i >= 0; i--) {
        result[i] = result[i] * suffix;
        suffix *= nums[i];
    }
    
    return result;
}

// Alternative implementation with separate arrays
function productExceptSelfSeparate(nums) {
    const n = nums.length;
    const prefix = new Array(n);
    const suffix = new Array(n);
    const result = new Array(n);
    
    // Calculate prefix products
    prefix[0] = 1;
    for (let i = 1; i < n; i++) {
        prefix[i] = prefix[i - 1] * nums[i - 1];
    }
    
    // Calculate suffix products
    suffix[n - 1] = 1;
    for (let i = n - 2; i >= 0; i--) {
        suffix[i] = suffix[i + 1] * nums[i + 1];
    }
    
    // Calculate result
    for (let i = 0; i < n; i++) {
        result[i] = prefix[i] * suffix[i];
    }
    
    return result;
}
```

### **Brute Force Solution**

```javascript
/**
 * Product of Array Except Self - Brute Force
 * Time Complexity: O(n²)
 * Space Complexity: O(1) - excluding output array
 * 
 * @param {number[]} nums
 * @return {number[]}
 */
function productExceptSelfBruteForce(nums) {
    const n = nums.length;
    const result = new Array(n);
    
    for (let i = 0; i < n; i++) {
        let product = 1;
        for (let j = 0; j < n; j++) {
            if (i !== j) {
                product *= nums[j];
            }
        }
        result[i] = product;
    }
    
    return result;
}

// Optimized brute force with early termination
function productExceptSelfBruteForceOptimized(nums) {
    const n = nums.length;
    const result = new Array(n);
    
    for (let i = 0; i < n; i++) {
        let product = 1;
        let hasZero = false;
        
        for (let j = 0; j < n; j++) {
            if (i !== j) {
                if (nums[j] === 0) {
                    hasZero = true;
                    break;
                }
                product *= nums[j];
            }
        }
        
        result[i] = hasZero ? 0 : product;
    }
    
    return result;
}
```

### **Division Approach (Not Recommended)**

```javascript
/**
 * Product of Array Except Self - Division Approach
 * Time Complexity: O(n)
 * Space Complexity: O(1) - excluding output array
 * 
 * @param {number[]} nums
 * @return {number[]}
 */
function productExceptSelfDivision(nums) {
    const n = nums.length;
    const result = new Array(n);
    
    // Count zeros
    let zeroCount = 0;
    let zeroIndex = -1;
    let totalProduct = 1;
    
    for (let i = 0; i < n; i++) {
        if (nums[i] === 0) {
            zeroCount++;
            zeroIndex = i;
        } else {
            totalProduct *= nums[i];
        }
    }
    
    // Handle different cases
    if (zeroCount > 1) {
        // More than one zero, all products are zero
        return new Array(n).fill(0);
    } else if (zeroCount === 1) {
        // One zero, only that position has non-zero product
        for (let i = 0; i < n; i++) {
            result[i] = i === zeroIndex ? totalProduct : 0;
        }
    } else {
        // No zeros, use division
        for (let i = 0; i < n; i++) {
            result[i] = totalProduct / nums[i];
        }
    }
    
    return result;
}
```

## 🧪 **Test Cases**

```javascript
// Test helper function
function test(actual, expected, testName) {
    const isEqual = JSON.stringify(actual) === JSON.stringify(expected);
    console.log(`${isEqual ? '✅' : '❌'} ${testName}`);
    if (!isEqual) {
        console.log(`Expected: ${JSON.stringify(expected)}`);
        console.log(`Actual: ${JSON.stringify(actual)}`);
    }
}

// Test cases
test(productExceptSelf([1,2,3,4]), [24,12,8,6], "Example 1");
test(productExceptSelf([-1,1,0,-3,3]), [0,0,9,0,0], "Example 2");
test(productExceptSelf([2,3,4,5]), [60,40,30,24], "Custom test 1");
test(productExceptSelf([1,0]), [0,1], "With zero");
test(productExceptSelf([0,0]), [0,0], "Two zeros");
test(productExceptSelf([1,1,1,1]), [1,1,1,1], "All ones");
test(productExceptSelf([2,2,2,2]), [8,8,8,8], "All twos");
test(productExceptSelf([1,2,3]), [6,3,2], "Three elements");
test(productExceptSelf([1]), [1], "Single element");
test(productExceptSelf([]), [], "Empty array");
```

## 📊 **Complexity Analysis**

### **Prefix and Suffix Products**
- **Time Complexity**: O(n) - Two passes through array
- **Space Complexity**: O(1) - Only using constant extra space (excluding output)
- **Best Case**: O(n) - Always need to process all elements
- **Worst Case**: O(n) - Same as best case

### **Brute Force**
- **Time Complexity**: O(n²) - Nested loops
- **Space Complexity**: O(1) - Only using constant extra space (excluding output)
- **Best Case**: O(n²) - Always check all pairs
- **Worst Case**: O(n²) - Same as best case

### **Division Approach**
- **Time Complexity**: O(n) - Single pass + division
- **Space Complexity**: O(1) - Only using constant extra space (excluding output)
- **Best Case**: O(n) - Always need to process all elements
- **Worst Case**: O(n) - Same as best case

## 🎯 **Key Insights**

1. **No Division**: Problem specifically asks for solution without division
2. **Prefix Products**: Calculate products from left to right
3. **Suffix Products**: Calculate products from right to left
4. **Space Optimization**: Use result array for prefix, variable for suffix
5. **Zero Handling**: Special case when array contains zeros

## 🔄 **Variations**

### **Product of Array Except Self - Follow Up**
```javascript
// What if you could use division?
function productExceptSelfWithDivision(nums) {
    const totalProduct = nums.reduce((acc, num) => acc * num, 1);
    return nums.map(num => totalProduct / num);
}

// Handle zeros with division
function productExceptSelfWithDivisionZeros(nums) {
    const zeroCount = nums.filter(num => num === 0).length;
    
    if (zeroCount > 1) {
        return new Array(nums.length).fill(0);
    } else if (zeroCount === 1) {
        const zeroIndex = nums.indexOf(0);
        const productWithoutZero = nums.reduce((acc, num, index) => 
            index === zeroIndex ? acc : acc * num, 1);
        return nums.map((_, index) => index === zeroIndex ? productWithoutZero : 0);
    } else {
        const totalProduct = nums.reduce((acc, num) => acc * num, 1);
        return nums.map(num => totalProduct / num);
    }
}
```

### **Maximum Product Subarray**
```javascript
// Related problem: Maximum Product Subarray
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

### **Product of Array Except Self - K Products**
```javascript
// Generalization: Product of array except K elements
function productExceptK(nums, k) {
    const n = nums.length;
    const result = new Array(n);
    
    // Calculate prefix products
    result[0] = 1;
    for (let i = 1; i < n; i++) {
        result[i] = result[i - 1] * nums[i - 1];
    }
    
    // Calculate suffix products
    let suffix = 1;
    for (let i = n - 1; i >= 0; i--) {
        result[i] = result[i] * suffix;
        suffix *= nums[i];
    }
    
    // For k > 1, we need more complex logic
    // This is a simplified version for k = 1
    return result;
}
```

## 🎓 **Interview Tips**

### **Google Interview**
- **Start with brute force**: Show understanding of the problem
- **Optimize step by step**: Explain the prefix/suffix approach
- **Handle edge cases**: Zeros, single element, empty array
- **Code quality**: Clean implementation with proper variable names

### **Meta Interview**
- **Think out loud**: Explain the mathematical insight
- **Visualize**: Draw the array and trace through the algorithm
- **Test thoroughly**: Walk through examples step by step
- **Consider variations**: What if we could use division?

### **Amazon Interview**
- **Real-world context**: How would this apply to recommendation systems?
- **Optimization**: Can we solve it in O(1) space?
- **Edge cases**: What if the array has negative numbers?
- **Production code**: Write robust, well-tested code

## 📚 **Related Problems**

- [**Two Sum**](TwoSum.md/) - Two numbers that sum to target
- [**Maximum Subarray**](MaximumSubarray.md/) - Kadane's algorithm
- [**Maximum Product Subarray**](../../../algorithms/Arrays/MaximumProductSubarray.md) - Product instead of sum
- [**Trapping Rain Water**](../../../algorithms/Arrays/TrappingRainWater.md) - Prefix/suffix pattern

## 🎉 **Summary**

Product of Array Except Self teaches:
- **Prefix and suffix products** technique
- **Space optimization** strategies
- **Mathematical insight** for array problems
- **Edge case handling** (zeros, single elements)

This problem is fundamental for understanding prefix/suffix patterns and appears in many variations!

---

**🚀 Ready to solve more array problems? Check out the next problem!**
