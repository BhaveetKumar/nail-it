# Longest Substring Without Repeating Characters

## Problem Statement

Given a string s, find the length of the longest substring without repeating characters.

**Example 1:**
```
Input: s = "abcabcbb"
Output: 3
Explanation: The longest substring without repeating characters is "abc", with length 3.
```

**Example 2:**
```
Input: s = "bbbbb"
Output: 1
Explanation: The longest substring without repeating characters is "b", with length 1.
```

**Example 3:**
```
Input: s = "pwwkew"
Output: 3
Explanation: The longest substring without repeating characters is "wke", with length 3.
```

## Approach

### Brute Force Approach
1. Generate all possible substrings
2. Check each substring for unique characters
3. Return the length of the longest valid substring

**Time Complexity:** O(n³) - Generate all substrings and check each
**Space Complexity:** O(min(m,n)) - Character set size

### Sliding Window Approach
1. Use two pointers (left and right) to define window
2. Expand window by moving right pointer
3. Shrink window when duplicate character found
4. Track maximum window size

**Time Complexity:** O(n) - Each character visited at most twice
**Space Complexity:** O(min(m,n)) - Character set size

## Solution

```javascript
/**
 * Find length of longest substring without repeating characters
 * @param {string} s - Input string
 * @return {number} - Length of longest substring without repeating characters
 */
function lengthOfLongestSubstring(s) {
    if (!s || s.length === 0) return 0;
    
    const charMap = new Map();
    let left = 0;
    let maxLength = 0;
    
    for (let right = 0; right < s.length; right++) {
        const char = s[right];
        
        // If character is already in window, move left pointer
        if (charMap.has(char) && charMap.get(char) >= left) {
            left = charMap.get(char) + 1;
        }
        
        // Update character position
        charMap.set(char, right);
        
        // Update maximum length
        maxLength = Math.max(maxLength, right - left + 1);
    }
    
    return maxLength;
}

// Alternative implementation using Set
function lengthOfLongestSubstringSet(s) {
    if (!s || s.length === 0) return 0;
    
    const charSet = new Set();
    let left = 0;
    let maxLength = 0;
    
    for (let right = 0; right < s.length; right++) {
        const char = s[right];
        
        // Shrink window until no duplicates
        while (charSet.has(char)) {
            charSet.delete(s[left]);
            left++;
        }
        
        // Add current character
        charSet.add(char);
        
        // Update maximum length
        maxLength = Math.max(maxLength, right - left + 1);
    }
    
    return maxLength;
}

// Return the actual longest substring
function longestSubstringNoRepeat(s) {
    if (!s || s.length === 0) return "";
    
    const charMap = new Map();
    let left = 0;
    let maxLength = 0;
    let maxStart = 0;
    
    for (let right = 0; right < s.length; right++) {
        const char = s[right];
        
        // If character is already in window, move left pointer
        if (charMap.has(char) && charMap.get(char) >= left) {
            left = charMap.get(char) + 1;
        }
        
        // Update character position
        charMap.set(char, right);
        
        // Update maximum length and start position
        if (right - left + 1 > maxLength) {
            maxLength = right - left + 1;
            maxStart = left;
        }
    }
    
    return s.substring(maxStart, maxStart + maxLength);
}

// Optimized version with array for ASCII characters
function lengthOfLongestSubstringOptimized(s) {
    if (!s || s.length === 0) return 0;
    
    const charIndex = new Array(128).fill(-1);
    let left = 0;
    let maxLength = 0;
    
    for (let right = 0; right < s.length; right++) {
        const charCode = s.charCodeAt(right);
        
        // If character is already in window, move left pointer
        if (charIndex[charCode] >= left) {
            left = charIndex[charCode] + 1;
        }
        
        // Update character position
        charIndex[charCode] = right;
        
        // Update maximum length
        maxLength = Math.max(maxLength, right - left + 1);
    }
    
    return maxLength;
}
```

## Dry Run

**Input:** s = "abcabcbb"

```
Initial: charMap = {}, left = 0, maxLength = 0

Step 1: right = 0, char = 'a'
        charMap = {'a': 0}
        maxLength = max(0, 0-0+1) = 1

Step 2: right = 1, char = 'b'
        charMap = {'a': 0, 'b': 1}
        maxLength = max(1, 1-0+1) = 2

Step 3: right = 2, char = 'c'
        charMap = {'a': 0, 'b': 1, 'c': 2}
        maxLength = max(2, 2-0+1) = 3

Step 4: right = 3, char = 'a'
        charMap.has('a') = true, charMap.get('a') = 0 >= left = 0
        left = 0 + 1 = 1
        charMap = {'a': 3, 'b': 1, 'c': 2}
        maxLength = max(3, 3-1+1) = 3

Step 5: right = 4, char = 'b'
        charMap.has('b') = true, charMap.get('b') = 1 >= left = 1
        left = 1 + 1 = 2
        charMap = {'a': 3, 'b': 4, 'c': 2}
        maxLength = max(3, 4-2+1) = 3

Step 6: right = 5, char = 'c'
        charMap.has('c') = true, charMap.get('c') = 2 >= left = 2
        left = 2 + 1 = 3
        charMap = {'a': 3, 'b': 4, 'c': 5}
        maxLength = max(3, 5-3+1) = 3

Step 7: right = 6, char = 'b'
        charMap.has('b') = true, charMap.get('b') = 4 >= left = 3
        left = 4 + 1 = 5
        charMap = {'a': 3, 'b': 6, 'c': 5}
        maxLength = max(3, 6-5+1) = 3

Step 8: right = 7, char = 'b'
        charMap.has('b') = true, charMap.get('b') = 6 >= left = 5
        left = 6 + 1 = 7
        charMap = {'a': 3, 'b': 7, 'c': 5}
        maxLength = max(3, 7-7+1) = 3

Result: maxLength = 3
```

## Complexity Analysis

- **Time Complexity:** O(n) - Each character visited at most twice
- **Space Complexity:** O(min(m,n)) - Character set size

## Alternative Solutions

### Brute Force Approach
```javascript
function lengthOfLongestSubstringBruteForce(s) {
    if (!s || s.length === 0) return 0;
    
    let maxLength = 0;
    
    for (let i = 0; i < s.length; i++) {
        for (let j = i; j < s.length; j++) {
            const substring = s.substring(i, j + 1);
            if (isUnique(substring)) {
                maxLength = Math.max(maxLength, substring.length);
            }
        }
    }
    
    return maxLength;
}

function isUnique(s) {
    const charSet = new Set();
    for (const char of s) {
        if (charSet.has(char)) {
            return false;
        }
        charSet.add(char);
    }
    return true;
}
```

### Using Two Pointers with Set
```javascript
function lengthOfLongestSubstringTwoPointers(s) {
    if (!s || s.length === 0) return 0;
    
    const charSet = new Set();
    let left = 0;
    let right = 0;
    let maxLength = 0;
    
    while (right < s.length) {
        if (!charSet.has(s[right])) {
            charSet.add(s[right]);
            maxLength = Math.max(maxLength, right - left + 1);
            right++;
        } else {
            charSet.delete(s[left]);
            left++;
        }
    }
    
    return maxLength;
}
```

### Using Array for ASCII Characters
```javascript
function lengthOfLongestSubstringASCII(s) {
    if (!s || s.length === 0) return 0;
    
    const charCount = new Array(128).fill(0);
    let left = 0;
    let right = 0;
    let maxLength = 0;
    
    while (right < s.length) {
        const charCode = s.charCodeAt(right);
        charCount[charCode]++;
        
        while (charCount[charCode] > 1) {
            charCount[s.charCodeAt(left)]--;
            left++;
        }
        
        maxLength = Math.max(maxLength, right - left + 1);
        right++;
    }
    
    return maxLength;
}
```

## Test Cases

```javascript
// Test cases
console.log(lengthOfLongestSubstring("abcabcbb")); // 3
console.log(lengthOfLongestSubstring("bbbbb")); // 1
console.log(lengthOfLongestSubstring("pwwkew")); // 3
console.log(lengthOfLongestSubstring("")); // 0
console.log(lengthOfLongestSubstring(" ")); // 1
console.log(lengthOfLongestSubstring("dvdf")); // 3

// Get actual substring
console.log(longestSubstringNoRepeat("abcabcbb")); // "abc"
console.log(longestSubstringNoRepeat("bbbbb")); // "b"
console.log(longestSubstringNoRepeat("pwwkew")); // "wke"

// Edge cases
console.log(lengthOfLongestSubstring("a")); // 1
console.log(lengthOfLongestSubstring("abcdef")); // 6
console.log(lengthOfLongestSubstring("aab")); // 2
```

## Key Insights

1. **Sliding Window**: Use two pointers to maintain window
2. **Character Tracking**: Use Map or Set to track characters in window
3. **Window Shrinking**: Move left pointer when duplicate found
4. **Efficient Lookup**: O(1) character lookup with Map/Set
5. **Single Pass**: Process string in O(n) time

## Related Problems

- [Longest Substring with At Most K Distinct Characters](LongestSubstringKDistinct.md/) - Variable constraint
- [Minimum Window Substring](MinimumWindowSubstring.md/) - Target substring
- [Longest Repeating Character Replacement](LongestRepeatingCharacterReplacement.md/) - With replacements
- [Fruit Into Baskets](FruitIntoBaskets.md/) - At most 2 distinct characters
- [Permutation in String](PermutationInString.md/) - Find permutation
