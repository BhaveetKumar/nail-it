# 🔢 Three Sum

> **Classic array problem using two pointers technique**

## 📋 **Problem Statement**

Given an integer array `nums`, return all the triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.

Notice that the solution set must not contain duplicate triplets.

## 🎯 **Examples**

```javascript
// Example 1
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Explanation: 
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
nums[1] + nums[1] + nums[3] = (-1) + 0 + 1 = 0.
The distinct triplets are [-1,0,1] and [-1,-1,2].

// Example 2
Input: nums = [0,1,1]
Output: []
Explanation: The only possible triplet does not sum up to 0.

// Example 3
Input: nums = [0,0,0]
Output: [[0,0,0]]
Explanation: The only possible triplet sums up to 0.
```

## 🧠 **Approach**

### **Brute Force Approach**
- Check all possible triplets
- Time Complexity: O(n³)
- Space Complexity: O(1)

### **Two Pointers Approach (Optimal)**
- Sort the array first
- Fix one element and use two pointers for the other two
- Skip duplicates to avoid duplicate triplets
- Time Complexity: O(n²)
- Space Complexity: O(1)

## 🔍 **Dry Run**

```
nums = [-1, 0, 1, 2, -1, -4]
sorted = [-4, -1, -1, 0, 1, 2]

i=0, nums[i]=-4, target=4
  left=1, right=5: nums[left]=-1, nums[right]=2, sum=1 < 4, left++
  left=2, right=5: nums[left]=-1, nums[right]=2, sum=1 < 4, left++
  left=3, right=5: nums[left]=0, nums[right]=2, sum=2 < 4, left++
  left=4, right=5: nums[left]=1, nums[right]=2, sum=3 < 4, left++
  left=5, right=5: break

i=1, nums[i]=-1, target=1
  left=2, right=5: nums[left]=-1, nums[right]=2, sum=1 == 1, found [-1,-1,2]
  left=3, right=4: nums[left]=0, nums[right]=1, sum=1 == 1, found [-1,0,1]
  left=4, right=4: break

i=2, nums[i]=-1 (duplicate), skip
i=3, nums[i]=0, target=0
  left=4, right=5: nums[left]=1, nums[right]=2, sum=3 > 0, right--
  left=4, right=4: break

Result: [[-1,-1,2], [-1,0,1]]
```

## 💻 **Solution**

### **Two Pointers Solution (Optimal)**

```javascript
/**
 * Three Sum - Two Pointers Approach
 * Time Complexity: O(n²)
 * Space Complexity: O(1)
 * 
 * @param {number[]} nums
 * @return {number[][]}
 */
function threeSum(nums) {
    const result = [];
    const n = nums.length;
    
    // Sort the array
    nums.sort((a, b) => a - b);
    
    for (let i = 0; i < n - 2; i++) {
        // Skip duplicates for the first element
        if (i > 0 && nums[i] === nums[i - 1]) {
            continue;
        }
        
        const target = -nums[i];
        let left = i + 1;
        let right = n - 1;
        
        while (left < right) {
            const sum = nums[left] + nums[right];
            
            if (sum === target) {
                result.push([nums[i], nums[left], nums[right]]);
                
                // Skip duplicates for the second element
                while (left < right && nums[left] === nums[left + 1]) {
                    left++;
                }
                
                // Skip duplicates for the third element
                while (left < right && nums[right] === nums[right - 1]) {
                    right--;
                }
                
                left++;
                right--;
            } else if (sum < target) {
                left++;
            } else {
                right--;
            }
        }
    }
    
    return result;
}

// Alternative implementation with Set for deduplication
function threeSumWithSet(nums) {
    const result = new Set();
    const n = nums.length;
    
    nums.sort((a, b) => a - b);
    
    for (let i = 0; i < n - 2; i++) {
        if (i > 0 && nums[i] === nums[i - 1]) continue;
        
        const target = -nums[i];
        let left = i + 1;
        let right = n - 1;
        
        while (left < right) {
            const sum = nums[left] + nums[right];
            
            if (sum === target) {
                result.add(JSON.stringify([nums[i], nums[left], nums[right]]));
                left++;
                right--;
            } else if (sum < target) {
                left++;
            } else {
                right--;
            }
        }
    }
    
    return Array.from(result).map(str => JSON.parse(str));
}
```

### **Brute Force Solution**

```javascript
/**
 * Three Sum - Brute Force Approach
 * Time Complexity: O(n³)
 * Space Complexity: O(1)
 * 
 * @param {number[]} nums
 * @return {number[][]}
 */
function threeSumBruteForce(nums) {
    const result = [];
    const n = nums.length;
    const seen = new Set();
    
    for (let i = 0; i < n - 2; i++) {
        for (let j = i + 1; j < n - 1; j++) {
            for (let k = j + 1; k < n; k++) {
                if (nums[i] + nums[j] + nums[k] === 0) {
                    const triplet = [nums[i], nums[j], nums[k]].sort((a, b) => a - b);
                    const key = JSON.stringify(triplet);
                    
                    if (!seen.has(key)) {
                        seen.add(key);
                        result.push(triplet);
                    }
                }
            }
        }
    }
    
    return result;
}
```

### **Hash Map Solution**

```javascript
/**
 * Three Sum - Hash Map Approach
 * Time Complexity: O(n²)
 * Space Complexity: O(n)
 * 
 * @param {number[]} nums
 * @return {number[][]}
 */
function threeSumHashMap(nums) {
    const result = [];
    const n = nums.length;
    const seen = new Set();
    
    for (let i = 0; i < n - 2; i++) {
        if (i > 0 && nums[i] === nums[i - 1]) continue;
        
        const target = -nums[i];
        const map = new Map();
        
        for (let j = i + 1; j < n; j++) {
            const complement = target - nums[j];
            
            if (map.has(complement)) {
                const triplet = [nums[i], complement, nums[j]].sort((a, b) => a - b);
                const key = JSON.stringify(triplet);
                
                if (!seen.has(key)) {
                    seen.add(key);
                    result.push(triplet);
                }
            }
            
            map.set(nums[j], j);
        }
    }
    
    return result;
}
```

## 🧪 **Test Cases**

```javascript
// Test helper function
function test(actual, expected, testName) {
    const isEqual = JSON.stringify(actual.sort()) === JSON.stringify(expected.sort());
    console.log(`${isEqual ? '✅' : '❌'} ${testName}`);
    if (!isEqual) {
        console.log(`Expected: ${JSON.stringify(expected)}`);
        console.log(`Actual: ${JSON.stringify(actual)}`);
    }
}

// Test cases
test(threeSum([-1,0,1,2,-1,-4]), [[-1,-1,2],[-1,0,1]], "Example 1");
test(threeSum([0,1,1]), [], "Example 2");
test(threeSum([0,0,0]), [[0,0,0]], "Example 3");
test(threeSum([-2,0,1,1,2]), [[-2,0,2],[-2,1,1]], "Custom test 1");
test(threeSum([1,2,-2,-1]), [], "No solution");
test(threeSum([-1,0,1,0]), [[-1,0,1]], "With duplicates");
```

## 📊 **Complexity Analysis**

### **Two Pointers Approach**
- **Time Complexity**: O(n²) - Nested loops with two pointers
- **Space Complexity**: O(1) - Only using constant extra space
- **Best Case**: O(n²) - Always need to check all pairs
- **Worst Case**: O(n²) - Same as best case

### **Brute Force Approach**
- **Time Complexity**: O(n³) - Three nested loops
- **Space Complexity**: O(1) - Only using constant extra space
- **Best Case**: O(n³) - Always check all triplets
- **Worst Case**: O(n³) - Same as best case

### **Hash Map Approach**
- **Time Complexity**: O(n²) - Two nested loops with hash map
- **Space Complexity**: O(n) - Hash map storage
- **Best Case**: O(n²) - Always need to check all pairs
- **Worst Case**: O(n²) - Same as best case

## 🎯 **Key Insights**

1. **Sorting First**: Sort array to enable two pointers technique
2. **Skip Duplicates**: Avoid duplicate triplets by skipping duplicates
3. **Two Pointers**: Use left and right pointers for efficient searching
4. **Target Calculation**: Calculate target as negative of current element
5. **Early Termination**: Can terminate early if current element > 0

## 🔄 **Variations**

### **Three Sum Closest**
```javascript
// Find three integers whose sum is closest to target
function threeSumClosest(nums, target) {
    nums.sort((a, b) => a - b);
    let closestSum = nums[0] + nums[1] + nums[2];
    
    for (let i = 0; i < nums.length - 2; i++) {
        let left = i + 1;
        let right = nums.length - 1;
        
        while (left < right) {
            const sum = nums[i] + nums[left] + nums[right];
            
            if (Math.abs(sum - target) < Math.abs(closestSum - target)) {
                closestSum = sum;
            }
            
            if (sum < target) {
                left++;
            } else {
                right--;
            }
        }
    }
    
    return closestSum;
}
```

### **Four Sum**
```javascript
// Find four integers that sum to target
function fourSum(nums, target) {
    const result = [];
    const n = nums.length;
    
    if (n < 4) return result;
    
    nums.sort((a, b) => a - b);
    
    for (let i = 0; i < n - 3; i++) {
        if (i > 0 && nums[i] === nums[i - 1]) continue;
        
        for (let j = i + 1; j < n - 2; j++) {
            if (j > i + 1 && nums[j] === nums[j - 1]) continue;
            
            let left = j + 1;
            let right = n - 1;
            
            while (left < right) {
                const sum = nums[i] + nums[j] + nums[left] + nums[right];
                
                if (sum === target) {
                    result.push([nums[i], nums[j], nums[left], nums[right]]);
                    
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
        }
    }
    
    return result;
}
```

### **Three Sum Smaller**
```javascript
// Count triplets with sum less than target
function threeSumSmaller(nums, target) {
    nums.sort((a, b) => a - b);
    let count = 0;
    
    for (let i = 0; i < nums.length - 2; i++) {
        let left = i + 1;
        let right = nums.length - 1;
        
        while (left < right) {
            if (nums[i] + nums[left] + nums[right] < target) {
                count += right - left;
                left++;
            } else {
                right--;
            }
        }
    }
    
    return count;
}
```

## 🎓 **Interview Tips**

### **Google Interview**
- **Start with brute force**: Show understanding of the problem
- **Optimize step by step**: Explain the two pointers approach
- **Handle edge cases**: Empty array, less than 3 elements, all zeros
- **Code quality**: Clean implementation with proper variable names

### **Meta Interview**
- **Think out loud**: Explain the sorting and two pointers approach
- **Visualize**: Draw the array and trace through the algorithm
- **Test thoroughly**: Walk through examples step by step
- **Consider variations**: What if we need the closest sum?

### **Amazon Interview**
- **Real-world context**: How would this apply to finding product combinations?
- **Optimization**: Can we solve it in O(1) space?
- **Edge cases**: What if the array has duplicates?
- **Production code**: Write robust, well-tested code

## 📚 **Related Problems**

- [**Two Sum**](TwoSum.md/) - Two numbers that sum to target
- [**Four Sum**](../../../algorithms/TwoPointers/FourSum.md) - Four numbers that sum to target
- [**Two Sum II**](TwoSumII.md/) - Sorted array two pointers
- [**3Sum Closest**](../../../algorithms/Arrays/ThreeSumClosest.md) - Closest sum to target

## 🎉 **Summary**

Three Sum teaches:
- **Two pointers technique** for sorted arrays
- **Duplicate handling** in array problems
- **Sorting optimization** for efficient searching
- **Nested loop optimization** with pointers

This problem is fundamental for understanding two pointers and appears in many variations!

---

**🚀 Ready to solve more array problems? Check out the next problem!**
