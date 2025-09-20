# 🔍 Pattern Matching - String Algorithms

## Problem Statement

Given a text string and a pattern string, find all occurrences of the pattern in the text.

## Examples

```javascript
// Example 1
Text: "ABABDABACDABABCABAB"
Pattern: "ABABCABAB"
Output: [10] // Pattern found at index 10

// Example 2
Text: "AABAACAADAABAABA"
Pattern: "AABA"
Output: [0, 9, 12] // Pattern found at indices 0, 9, and 12
```

## Approaches

### 1. Naive Pattern Matching (Brute Force)
- **Time**: O(m × n) where m = text length, n = pattern length
- **Space**: O(1)

### 2. KMP (Knuth-Morris-Pratt) Algorithm
- **Time**: O(m + n)
- **Space**: O(n)

### 3. Rabin-Karp Algorithm
- **Time**: O(m + n) average case, O(m × n) worst case
- **Space**: O(1)

## Solutions

### Approach 1: Naive Pattern Matching

```javascript
/**
 * Naive pattern matching algorithm
 * @param {string} text
 * @param {string} pattern
 * @return {number[]}
 */
function naivePatternMatching(text, pattern) {
  const result = [];
  const textLen = text.length;
  const patternLen = pattern.length;
  
  // Slide pattern over text
  for (let i = 0; i <= textLen - patternLen; i++) {
    let j = 0;
    
    // Check if pattern matches at current position
    while (j < patternLen && text[i + j] === pattern[j]) {
      j++;
    }
    
    // If pattern matched completely
    if (j === patternLen) {
      result.push(i);
    }
  }
  
  return result;
}
```

### Approach 2: KMP Algorithm

```javascript
/**
 * KMP (Knuth-Morris-Pratt) algorithm
 * @param {string} text
 * @param {string} pattern
 * @return {number[]}
 */
function kmpPatternMatching(text, pattern) {
  const result = [];
  const textLen = text.length;
  const patternLen = pattern.length;
  
  // Build LPS (Longest Proper Prefix which is also Suffix) array
  const lps = buildLPS(pattern);
  
  let i = 0; // Index for text
  let j = 0; // Index for pattern
  
  while (i < textLen) {
    if (pattern[j] === text[i]) {
      i++;
      j++;
    }
    
    if (j === patternLen) {
      result.push(i - j);
      j = lps[j - 1];
    } else if (i < textLen && pattern[j] !== text[i]) {
      if (j !== 0) {
        j = lps[j - 1];
      } else {
        i++;
      }
    }
  }
  
  return result;
}

/**
 * Build Longest Proper Prefix which is also Suffix array
 * @param {string} pattern
 * @return {number[]}
 */
function buildLPS(pattern) {
  const lps = new Array(pattern.length).fill(0);
  let len = 0; // Length of the previous longest prefix suffix
  let i = 1;
  
  while (i < pattern.length) {
    if (pattern[i] === pattern[len]) {
      len++;
      lps[i] = len;
      i++;
    } else {
      if (len !== 0) {
        len = lps[len - 1];
      } else {
        lps[i] = 0;
        i++;
      }
    }
  }
  
  return lps;
}
```

### Approach 3: Rabin-Karp Algorithm

```javascript
/**
 * Rabin-Karp algorithm using rolling hash
 * @param {string} text
 * @param {string} pattern
 * @return {number[]}
 */
function rabinKarpPatternMatching(text, pattern) {
  const result = [];
  const textLen = text.length;
  const patternLen = pattern.length;
  const base = 256; // Base for hash function
  const mod = 1000000007; // Large prime number
  
  // Calculate hash of pattern and first window of text
  let patternHash = 0;
  let textHash = 0;
  let h = 1; // h = base^(patternLen-1) % mod
  
  // Calculate h
  for (let i = 0; i < patternLen - 1; i++) {
    h = (h * base) % mod;
  }
  
  // Calculate initial hash values
  for (let i = 0; i < patternLen; i++) {
    patternHash = (base * patternHash + pattern.charCodeAt(i)) % mod;
    textHash = (base * textHash + text.charCodeAt(i)) % mod;
  }
  
  // Slide the pattern over text
  for (let i = 0; i <= textLen - patternLen; i++) {
    // Check if hash values match
    if (patternHash === textHash) {
      // Verify character by character to avoid hash collisions
      let j = 0;
      while (j < patternLen && text[i + j] === pattern[j]) {
        j++;
      }
      
      if (j === patternLen) {
        result.push(i);
      }
    }
    
    // Calculate hash for next window
    if (i < textLen - patternLen) {
      textHash = (base * (textHash - text.charCodeAt(i) * h) + text.charCodeAt(i + patternLen)) % mod;
      
      // Handle negative hash values
      if (textHash < 0) {
        textHash += mod;
      }
    }
  }
  
  return result;
}
```

### Approach 4: Boyer-Moore Algorithm (Simplified)

```javascript
/**
 * Simplified Boyer-Moore algorithm
 * @param {string} text
 * @param {string} pattern
 * @return {number[]}
 */
function boyerMoorePatternMatching(text, pattern) {
  const result = [];
  const textLen = text.length;
  const patternLen = pattern.length;
  
  // Build bad character table
  const badChar = buildBadCharTable(pattern);
  
  let shift = 0;
  
  while (shift <= textLen - patternLen) {
    let j = patternLen - 1;
    
    // Keep reducing index j while characters match
    while (j >= 0 && pattern[j] === text[shift + j]) {
      j--;
    }
    
    if (j < 0) {
      // Pattern found
      result.push(shift);
      
      // Shift pattern to next occurrence
      shift += (shift + patternLen < textLen) ? 
        patternLen - (badChar[text.charCodeAt(shift + patternLen)] || -1) : 1;
    } else {
      // Shift pattern based on bad character rule
      shift += Math.max(1, j - (badChar[text.charCodeAt(shift + j)] || -1));
    }
  }
  
  return result;
}

/**
 * Build bad character table for Boyer-Moore
 * @param {string} pattern
 * @return {Object}
 */
function buildBadCharTable(pattern) {
  const badChar = {};
  
  for (let i = 0; i < pattern.length; i++) {
    badChar[pattern.charCodeAt(i)] = i;
  }
  
  return badChar;
}
```

## Advanced Pattern Matching

### Wildcard Pattern Matching

```javascript
/**
 * Wildcard pattern matching with * and ?
 * @param {string} text
 * @param {string} pattern
 * @return {boolean}
 */
function wildcardPatternMatching(text, pattern) {
  const textLen = text.length;
  const patternLen = pattern.length;
  
  // Create DP table
  const dp = Array(textLen + 1).fill().map(() => Array(patternLen + 1).fill(false));
  
  // Empty pattern matches empty text
  dp[0][0] = true;
  
  // Handle patterns starting with *
  for (let j = 1; j <= patternLen; j++) {
    if (pattern[j - 1] === '*') {
      dp[0][j] = dp[0][j - 1];
    }
  }
  
  // Fill DP table
  for (let i = 1; i <= textLen; i++) {
    for (let j = 1; j <= patternLen; j++) {
      if (pattern[j - 1] === '*') {
        // * can match 0 or more characters
        dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
      } else if (pattern[j - 1] === '?' || text[i - 1] === pattern[j - 1]) {
        // ? matches any character or characters match
        dp[i][j] = dp[i - 1][j - 1];
      }
    }
  }
  
  return dp[textLen][patternLen];
}
```

### Regular Expression Matching

```javascript
/**
 * Regular expression matching with . and *
 * @param {string} text
 * @param {string} pattern
 * @return {boolean}
 */
function regexPatternMatching(text, pattern) {
  const textLen = text.length;
  const patternLen = pattern.length;
  
  // Create DP table
  const dp = Array(textLen + 1).fill().map(() => Array(patternLen + 1).fill(false));
  
  // Empty pattern matches empty text
  dp[0][0] = true;
  
  // Handle patterns like a*, a*b*, etc.
  for (let j = 2; j <= patternLen; j += 2) {
    if (pattern[j - 1] === '*') {
      dp[0][j] = dp[0][j - 2];
    }
  }
  
  // Fill DP table
  for (let i = 1; i <= textLen; i++) {
    for (let j = 1; j <= patternLen; j++) {
      if (pattern[j - 1] === '*') {
        // * can match 0 or more of previous character
        dp[i][j] = dp[i][j - 2]; // Match 0 characters
        
        if (pattern[j - 2] === '.' || pattern[j - 2] === text[i - 1]) {
          dp[i][j] = dp[i][j] || dp[i - 1][j]; // Match 1 or more characters
        }
      } else if (pattern[j - 1] === '.' || pattern[j - 1] === text[i - 1]) {
        dp[i][j] = dp[i - 1][j - 1];
      }
    }
  }
  
  return dp[textLen][patternLen];
}
```

## Test Cases

```javascript
// Test cases
console.log("=== Pattern Matching Test Cases ===");

// Test 1
console.log("Test 1:");
const text1 = "ABABDABACDABABCABAB";
const pattern1 = "ABABCABAB";
console.log("Text:", text1);
console.log("Pattern:", pattern1);
console.log("Naive:", naivePatternMatching(text1, pattern1));
console.log("KMP:", kmpPatternMatching(text1, pattern1));
console.log("Rabin-Karp:", rabinKarpPatternMatching(text1, pattern1));
console.log("Boyer-Moore:", boyerMoorePatternMatching(text1, pattern1));
console.log();

// Test 2
console.log("Test 2:");
const text2 = "AABAACAADAABAABA";
const pattern2 = "AABA";
console.log("Text:", text2);
console.log("Pattern:", pattern2);
console.log("All algorithms:", naivePatternMatching(text2, pattern2));
console.log();

// Test 3
console.log("Test 3:");
const text3 = "GEEKSFORGEEKS";
const pattern3 = "GEEKS";
console.log("Text:", text3);
console.log("Pattern:", pattern3);
console.log("All algorithms:", naivePatternMatching(text3, pattern3));
console.log();

// Performance comparison
console.log("=== Performance Comparison ===");
const longText = "A".repeat(10000) + "B" + "A".repeat(10000);
const longPattern = "A".repeat(100) + "B";

console.log("Testing with long text (20,001 chars) and pattern (101 chars):");

let start = performance.now();
let result1 = naivePatternMatching(longText, longPattern);
let end = performance.now();
console.log(`Naive: ${result1.length} matches - Time: ${end - start}ms`);

start = performance.now();
let result2 = kmpPatternMatching(longText, longPattern);
end = performance.now();
console.log(`KMP: ${result2.length} matches - Time: ${end - start}ms`);

start = performance.now();
let result3 = rabinKarpPatternMatching(longText, longPattern);
end = performance.now();
console.log(`Rabin-Karp: ${result3.length} matches - Time: ${end - start}ms`);

// Wildcard matching
console.log("\n=== Wildcard Pattern Matching ===");
console.log("Text: 'adceb', Pattern: '*a*b'");
console.log("Result:", wildcardPatternMatching("adceb", "*a*b"));

console.log("Text: 'acdcb', Pattern: 'a*c?b'");
console.log("Result:", wildcardPatternMatching("acdcb", "a*c?b"));

// Regex matching
console.log("\n=== Regular Expression Matching ===");
console.log("Text: 'aa', Pattern: 'a*'");
console.log("Result:", regexPatternMatching("aa", "a*"));

console.log("Text: 'ab', Pattern: '.*'");
console.log("Result:", regexPatternMatching("ab", ".*"));
```

## Visualization

```javascript
/**
 * Visualize pattern matching process
 * @param {string} text
 * @param {string} pattern
 * @param {number} position
 */
function visualizePatternMatching(text, pattern, position) {
  console.log("Pattern Matching Visualization:");
  console.log("Text:    ", text);
  console.log("Pattern: ", " ".repeat(position) + pattern);
  console.log("Match:   ", position >= 0 ? "✓" : "✗");
  if (position >= 0) {
    console.log(`Found at index: ${position}`);
  }
}

// Example visualization
console.log("=== Visualization ===");
const exampleText = "ABABDABACDABABCABAB";
const examplePattern = "ABABCABAB";
const matches = kmpPatternMatching(exampleText, examplePattern);

matches.forEach(match => {
  visualizePatternMatching(exampleText, examplePattern, match);
  console.log();
});
```

## Key Insights

1. **Naive Approach**: Simple but inefficient for large inputs
2. **KMP Algorithm**: Uses preprocessing to avoid redundant comparisons
3. **Rabin-Karp**: Uses hashing for average O(m+n) performance
4. **Boyer-Moore**: Skips characters using bad character rule
5. **Wildcard Matching**: Requires dynamic programming for complex patterns
6. **Regex Matching**: Handles special characters like . and *

## Common Mistakes

1. **Off-by-one errors** in array indexing
2. **Not handling edge cases** (empty strings, single characters)
3. **Incorrect LPS calculation** in KMP algorithm
4. **Hash collision handling** in Rabin-Karp
5. **Incorrect shift calculation** in Boyer-Moore

## Related Problems

- [Find All Anagrams in String](../../../algorithms/SlidingWindow/FindAllAnagramsInString.md)
- [Longest Common Subsequence](../../../algorithms/DynamicProgramming/LongestCommonSubsequence.md)
- [Edit Distance](../../../algorithms/DynamicProgramming/EditDistance.md)
- [Palindrome Substrings](PalindromeSubstrings.md)

## Interview Tips

1. **Start with naive approach** and explain its limitations
2. **Implement KMP algorithm** as it's the most commonly asked
3. **Explain the LPS array** and its purpose
4. **Discuss time/space complexity** trade-offs
5. **Handle edge cases** properly
6. **Consider multiple pattern matching** extensions
