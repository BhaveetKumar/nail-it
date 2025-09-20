# Sliding Window Problems

The sliding window technique is used to solve problems involving arrays or strings where we need to find a subarray or substring that satisfies certain conditions. It's particularly useful for problems involving:

- **Fixed Size Windows**: Find optimal subarray of fixed size
- **Variable Size Windows**: Find optimal subarray of variable size
- **String Problems**: Find substrings with specific properties
- **Optimization Problems**: Find maximum/minimum subarray
- **Pattern Matching**: Find patterns in sequences

## Key Concepts

### Sliding Window Template
```javascript
function slidingWindow(arr, k) {
    let left = 0;
    let result = 0;
    let windowSum = 0;
    
    // Expand window
    for (let right = 0; right < arr.length; right++) {
        windowSum += arr[right];
        
        // Shrink window if needed
        while (right - left + 1 > k) {
            windowSum -= arr[left];
            left++;
        }
        
        // Process window
        if (right - left + 1 === k) {
            result = Math.max(result, windowSum);
        }
    }
    
    return result;
}
```

### Common Patterns
1. **Fixed Window**: Window size is constant
2. **Variable Window**: Window size changes based on conditions
3. **Two Pointers**: Left and right pointers move independently
4. **Prefix Sum**: Use prefix sum for efficient window calculations
5. **Hash Map**: Track character/element frequencies

## Problems

### 1. [Maximum Sum Subarray of Size K](MaximumSumSubarrayK.md/)
Find the maximum sum of any contiguous subarray of size k.

### 2. [Longest Substring Without Repeating Characters](LongestSubstringNoRepeat.md/)
Find the length of the longest substring without repeating characters.

### 3. [Minimum Window Substring](../../../algorithms/SlidingWindow/MinimumWindowSubstring.md)
Find the minimum window in string S that contains all characters in string T.

### 4. [Longest Substring with At Most K Distinct Characters](LongestSubstringKDistinct.md/)
Find the length of the longest substring with at most k distinct characters.

### 5. [Fruit Into Baskets](../../../algorithms/SlidingWindow/FruitIntoBaskets.md)
Find the maximum number of fruits you can collect with two baskets.

### 6. [Longest Repeating Character Replacement](../../../algorithms/SlidingWindow/LongestRepeatingCharacterReplacement.md)
Find the length of the longest substring with same character after k replacements.

### 7. [Permutation in String](../../../algorithms/Strings/PermutationInString.md)
Check if string s2 contains a permutation of string s1.

### 8. [Find All Anagrams in a String](FindAllAnagrams.md/)
Find all anagrams of string p in string s.

### 9. [Subarray Product Less Than K](SubarrayProductLessThanK.md/)
Count the number of contiguous subarrays where product is less than k.

### 10. [Maximum Points You Can Obtain from Cards](MaximumPointsFromCards.md/)
Find maximum points by taking k cards from either end.

## Time & Space Complexity

| Problem | Time Complexity | Space Complexity |
|---------|----------------|------------------|
| Maximum Sum Subarray K | O(n) | O(1) |
| Longest Substring No Repeat | O(n) | O(min(m,n)) |
| Minimum Window Substring | O(|s| + |t|) | O(|s| + |t|) |
| Longest Substring K Distinct | O(n) | O(k) |
| Fruit Into Baskets | O(n) | O(1) |
| Longest Repeating Replacement | O(n) | O(1) |
| Permutation in String | O(|s1| + |s2|) | O(1) |
| Find All Anagrams | O(|s| + |p|) | O(1) |
| Subarray Product Less Than K | O(n) | O(1) |
| Maximum Points From Cards | O(k) | O(1) |

Where:
- n = array/string length
- m = character set size
- k = window size or constraint

## Tips for Sliding Window Problems

1. **Identify Window Type**: Fixed or variable size window
2. **Define Window State**: What information to track in the window
3. **Expand Window**: Move right pointer to include new elements
4. **Shrink Window**: Move left pointer to maintain window properties
5. **Update Result**: Process the current window state
6. **Handle Edge Cases**: Empty arrays, single elements, etc.

## Common Mistakes

1. **Incorrect Window Size**: Not maintaining proper window boundaries
2. **Missing Updates**: Not updating window state when expanding/shrinking
3. **Off-by-One Errors**: Incorrect pointer movements
4. **State Management**: Not properly tracking window properties
5. **Edge Cases**: Not handling empty inputs or single elements
