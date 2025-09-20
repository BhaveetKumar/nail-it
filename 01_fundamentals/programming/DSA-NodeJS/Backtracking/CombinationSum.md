# Combination Sum

## Problem Statement

Given an array of distinct integers `candidates` and a target integer `target`, return a list of all unique combinations of `candidates` where the chosen numbers sum to `target`. You may return the combinations in any order.

The same number may be chosen from `candidates` an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

**Example 1:**
```
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.
```

**Example 2:**
```
Input: candidates = [2,3,5], target = 8
Output: [[2,2,2,2],[2,3,3],[3,5]]
```

**Example 3:**
```
Input: candidates = [2], target = 1
Output: []
```

## Approach

### Backtracking Approach
1. Sort the candidates array for optimization
2. Use backtracking to explore all possible combinations
3. For each candidate, try including it in the current combination
4. If the sum equals target, add to result
5. If sum exceeds target, backtrack
6. Continue with next candidates

**Time Complexity:** O(2^target) - In the worst case, we explore all possible combinations
**Space Complexity:** O(target) - Maximum recursion depth

## Solution

```javascript
/**
 * Find all unique combinations that sum to target
 * @param {number[]} candidates - Array of distinct integers
 * @param {number} target - Target sum
 * @return {number[][]} - All unique combinations
 */
function combinationSum(candidates, target) {
    const result = [];
    
    // Sort candidates for optimization
    candidates.sort((a, b) => a - b);
    
    function backtrack(start, current, sum) {
        // Base case: found a valid combination
        if (sum === target) {
            result.push([...current]);
            return;
        }
        
        // Try each candidate starting from 'start' index
        for (let i = start; i < candidates.length; i++) {
            const num = candidates[i];
            
            // Pruning: if adding this number exceeds target, skip
            if (sum + num > target) {
                break;
            }
            
            // Add current number to combination
            current.push(num);
            
            // Recurse with same index (can reuse same number)
            backtrack(i, current, sum + num);
            
            // Backtrack: remove the number
            current.pop();
        }
    }
    
    backtrack(0, [], 0);
    return result;
}

// Alternative implementation with explicit sum tracking
function combinationSumAlternative(candidates, target) {
    const result = [];
    candidates.sort((a, b) => a - b);
    
    function backtrack(index, current, remaining) {
        if (remaining === 0) {
            result.push([...current]);
            return;
        }
        
        for (let i = index; i < candidates.length; i++) {
            const num = candidates[i];
            
            if (num > remaining) {
                break;
            }
            
            current.push(num);
            backtrack(i, current, remaining - num);
            current.pop();
        }
    }
    
    backtrack(0, [], target);
    return result;
}
```

## Dry Run

**Input:** candidates = [2,3,6,7], target = 7

```
Sorted candidates: [2,3,6,7]

Initial: start = 0, current = [], sum = 0

1. i = 0, num = 2
   current = [2], sum = 2
   sum + 2 = 4 <= 7, continue
   
   1.1. i = 0, num = 2
        current = [2,2], sum = 4
        sum + 2 = 6 <= 7, continue
        
        1.1.1. i = 0, num = 2
               current = [2,2,2], sum = 6
               sum + 2 = 8 > 7, break
               
        1.1.2. i = 1, num = 3
               current = [2,2,3], sum = 7
               sum === target, add to result: [[2,2,3]]
               Backtrack: current = [2,2]
               
        1.1.3. i = 2, num = 6
               current = [2,2,6], sum = 10
               sum > target, break
               
        1.1.4. i = 3, num = 7
               current = [2,2,7], sum = 11
               sum > target, break
               
        Backtrack: current = [2]
        
   1.2. i = 1, num = 3
        current = [2,3], sum = 5
        sum + 3 = 8 > 7, break
        
   1.3. i = 2, num = 6
        current = [2,6], sum = 8
        sum > target, break
        
   1.4. i = 3, num = 7
        current = [2,7], sum = 9
        sum > target, break
        
    Backtrack: current = []
    
2. i = 1, num = 3
   current = [3], sum = 3
   sum + 3 = 6 <= 7, continue
   
   2.1. i = 1, num = 3
        current = [3,3], sum = 6
        sum + 3 = 9 > 7, break
        
   2.2. i = 2, num = 6
        current = [3,6], sum = 9
        sum > target, break
        
   2.3. i = 3, num = 7
        current = [3,7], sum = 10
        sum > target, break
        
    Backtrack: current = []
    
3. i = 2, num = 6
   current = [6], sum = 6
   sum + 6 = 12 > 7, break
   
4. i = 3, num = 7
   current = [7], sum = 7
   sum === target, add to result: [[2,2,3], [7]]

Result: [[2,2,3], [7]]
```

## Complexity Analysis

- **Time Complexity:** O(2^target) - In the worst case, we explore all possible combinations
- **Space Complexity:** O(target) - Maximum recursion depth and current combination size

## Alternative Solutions

### Iterative Approach
```javascript
function combinationSumIterative(candidates, target) {
    const result = [];
    const stack = [{ index: 0, current: [], sum: 0 }];
    
    candidates.sort((a, b) => a - b);
    
    while (stack.length > 0) {
        const { index, current, sum } = stack.pop();
        
        if (sum === target) {
            result.push([...current]);
            continue;
        }
        
        for (let i = index; i < candidates.length; i++) {
            const num = candidates[i];
            
            if (sum + num > target) {
                break;
            }
            
            stack.push({
                index: i,
                current: [...current, num],
                sum: sum + num
            });
        }
    }
    
    return result;
}
```

### Dynamic Programming Approach
```javascript
function combinationSumDP(candidates, target) {
    const dp = Array(target + 1).fill().map(() => []);
    dp[0] = [[]];
    
    for (let i = 1; i <= target; i++) {
        for (const num of candidates) {
            if (i >= num) {
                for (const combination of dp[i - num]) {
                    dp[i].push([...combination, num]);
                }
            }
        }
    }
    
    return dp[target];
}
```

## Test Cases

```javascript
// Test cases
console.log(combinationSum([2,3,6,7], 7)); // [[2,2,3],[7]]
console.log(combinationSum([2,3,5], 8)); // [[2,2,2,2],[2,3,3],[3,5]]
console.log(combinationSum([2], 1)); // []
console.log(combinationSum([1], 1)); // [[1]]
console.log(combinationSum([1], 2)); // [[1,1]]

// Edge cases
console.log(combinationSum([], 1)); // []
console.log(combinationSum([1,2,3], 0)); // [[]]
```

## Key Insights

1. **Unlimited Reuse**: Same number can be used multiple times
2. **Sorting Optimization**: Sort candidates to enable early pruning
3. **Index Tracking**: Use start index to avoid duplicate combinations
4. **Pruning**: Skip candidates that would exceed target
5. **Backtracking**: Remove elements when backtracking

## Related Problems

- [Combination Sum II](../../../algorithms/Backtracking/CombinationSumII.md) - Each number used once
- [Combination Sum III](../../../algorithms/Backtracking/CombinationSumIII.md) - Use k numbers only
- [Combination Sum IV](CombinationSumIV.md/) - Count combinations
- [Subsets](../../../algorithms/BitManipulation/Subsets.md) - Generate all subsets
- [Permutations](../../../algorithms/Backtracking/Permutations.md) - Generate all permutations
