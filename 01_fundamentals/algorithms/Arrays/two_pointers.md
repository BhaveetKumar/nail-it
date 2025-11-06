---
# Auto-generated front matter
Title: Two Pointers
LastUpdated: 2025-11-06T20:45:58.720954
Tags: []
Status: draft
---

# 👆 **Two Pointers Technique**

## 📘 **Theory**

The Two Pointers technique is a fundamental algorithmic pattern that uses two pointers to traverse an array or sequence from different positions, typically from both ends or at different speeds. This technique is highly efficient for solving problems involving sorted arrays, palindromes, and finding pairs that meet certain criteria.

### **Why Two Pointers Matters**

- **Efficiency**: Reduces time complexity from O(n²) to O(n) for many problems
- **Space Optimization**: Often achieves O(1) space complexity
- **Intuitive**: Mirrors human problem-solving approach
- **Versatile**: Applicable to arrays, strings, linked lists, and other sequences
- **Interview Favorite**: Commonly asked in technical interviews

### **Common Pitfalls and Best Practices**

- **Pointer Movement**: Ensure pointers move in the correct direction
- **Boundary Conditions**: Handle edge cases when pointers meet or cross
- **Sorted Arrays**: Two pointers work best on sorted data
- **Duplicate Handling**: Consider how to handle duplicate elements
- **Index Management**: Be careful with array bounds and index calculations

## 📊 **Diagrams**

### **Two Pointers from Ends**

```
Array: [1, 2, 3, 4, 5, 6, 7, 8, 9]
       ↑                           ↑
    left                        right

Step 1: Check sum of left + right
Step 2: Move pointers based on comparison
Step 3: Continue until left >= right
```

### **Fast and Slow Pointers (Floyd's Cycle Detection)**

```
Linked List: 1 -> 2 -> 3 -> 4 -> 5 -> 3 (cycle)
             ↑    ↑
           slow fast

Step 1: Move slow by 1, fast by 2
Step 2: If they meet, cycle exists
Step 3: Find cycle start if needed
```

### **Sliding Window with Two Pointers**

```
Array: [1, 2, 3, 4, 5, 6, 7, 8, 9]
       ↑  ↑
    left right

Window expands/contracts based on condition
```

## 🧩 **Example**

**Problem**: Find two numbers in a sorted array that sum to a target value

**Input**: `[2, 7, 11, 15]`, target = 9
**Expected Output**: `[0, 1]` (indices of 2 and 7)

**Step-by-step**:

1. Initialize left pointer at start (0), right at end (3)
2. Check sum: arr[0] + arr[3] = 2 + 15 = 17 > 9
3. Move right pointer left: right = 2
4. Check sum: arr[0] + arr[2] = 2 + 11 = 13 > 9
5. Move right pointer left: right = 1
6. Check sum: arr[0] + arr[1] = 2 + 7 = 9 = target
7. Return indices [0, 1]

## 💻 **Implementation (Golang)**

```go
package main

import (
    "fmt"
    "sort"
)

// Two Sum - Find indices of two numbers that sum to target
func twoSum(nums []int, target int) []int {
    // Create a map to store value -> index
    numMap := make(map[int]int)

    for i, num := range nums {
        complement := target - num
        if index, exists := numMap[complement]; exists {
            return []int{index, i}
        }
        numMap[num] = i
    }

    return []int{} // No solution found
}

// Two Sum II - Sorted array (Two Pointers approach)
func twoSumSorted(nums []int, target int) []int {
    left, right := 0, len(nums)-1

    for left < right {
        sum := nums[left] + nums[right]

        if sum == target {
            return []int{left, right}
        } else if sum < target {
            left++
        } else {
            right--
        }
    }

    return []int{} // No solution found
}

// Three Sum - Find all unique triplets that sum to zero
func threeSum(nums []int) [][]int {
    sort.Ints(nums)
    var result [][]int

    for i := 0; i < len(nums)-2; i++ {
        // Skip duplicates for first number
        if i > 0 && nums[i] == nums[i-1] {
            continue
        }

        left, right := i+1, len(nums)-1
        target := -nums[i]

        for left < right {
            sum := nums[left] + nums[right]

            if sum == target {
                result = append(result, []int{nums[i], nums[left], nums[right]})

                // Skip duplicates for second number
                for left < right && nums[left] == nums[left+1] {
                    left++
                }
                // Skip duplicates for third number
                for left < right && nums[right] == nums[right-1] {
                    right--
                }

                left++
                right--
            } else if sum < target {
                left++
            } else {
                right--
            }
        }
    }

    return result
}

// Container With Most Water
func maxArea(height []int) int {
    left, right := 0, len(height)-1
    maxWater := 0

    for left < right {
        // Calculate current area
        width := right - left
        currentHeight := min(height[left], height[right])
        currentArea := width * currentHeight

        if currentArea > maxWater {
            maxWater = currentArea
        }

        // Move pointer with smaller height
        if height[left] < height[right] {
            left++
        } else {
            right--
        }
    }

    return maxWater
}

// Remove Duplicates from Sorted Array
func removeDuplicates(nums []int) int {
    if len(nums) == 0 {
        return 0
    }

    slow := 0
    for fast := 1; fast < len(nums); fast++ {
        if nums[fast] != nums[slow] {
            slow++
            nums[slow] = nums[fast]
        }
    }

    return slow + 1
}

// Move Zeroes to End
func moveZeroes(nums []int) {
    slow := 0

    // Move all non-zero elements to the front
    for fast := 0; fast < len(nums); fast++ {
        if nums[fast] != 0 {
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow++
        }
    }
}

// Valid Palindrome
func isPalindrome(s string) bool {
    left, right := 0, len(s)-1

    for left < right {
        // Skip non-alphanumeric characters
        for left < right && !isAlphanumeric(s[left]) {
            left++
        }
        for left < right && !isAlphanumeric(s[right]) {
            right--
        }

        // Compare characters (case insensitive)
        if toLowerCase(s[left]) != toLowerCase(s[right]) {
            return false
        }

        left++
        right--
    }

    return true
}

func isAlphanumeric(c byte) bool {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
}

func toLowerCase(c byte) byte {
    if c >= 'A' && c <= 'Z' {
        return c + 32
    }
    return c
}

// Trapping Rain Water
func trap(height []int) int {
    if len(height) < 3 {
        return 0
    }

    left, right := 0, len(height)-1
    leftMax, rightMax := 0, 0
    water := 0

    for left < right {
        if height[left] < height[right] {
            if height[left] >= leftMax {
                leftMax = height[left]
            } else {
                water += leftMax - height[left]
            }
            left++
        } else {
            if height[right] >= rightMax {
                rightMax = height[right]
            } else {
                water += rightMax - height[right]
            }
            right--
        }
    }

    return water
}

// Find All Anagrams in String
func findAnagrams(s string, p string) []int {
    if len(s) < len(p) {
        return []int{}
    }

    var result []int
    pCount := make(map[byte]int)
    windowCount := make(map[byte]int)

    // Count characters in pattern
    for i := 0; i < len(p); i++ {
        pCount[p[i]]++
    }

    // Sliding window
    left := 0
    for right := 0; right < len(s); right++ {
        // Add character to window
        windowCount[s[right]]++

        // If window size exceeds pattern length, remove left character
        if right-left+1 > len(p) {
            windowCount[s[left]]--
            if windowCount[s[left]] == 0 {
                delete(windowCount, s[left])
            }
            left++
        }

        // Check if current window is an anagram
        if right-left+1 == len(p) && mapsEqual(pCount, windowCount) {
            result = append(result, left)
        }
    }

    return result
}

func mapsEqual(m1, m2 map[byte]int) bool {
    if len(m1) != len(m2) {
        return false
    }

    for k, v := range m1 {
        if m2[k] != v {
            return false
        }
    }

    return true
}

// Longest Substring Without Repeating Characters
func lengthOfLongestSubstring(s string) int {
    charMap := make(map[byte]int)
    left, maxLen := 0, 0

    for right := 0; right < len(s); right++ {
        if index, exists := charMap[s[right]]; exists && index >= left {
            left = index + 1
        }

        charMap[s[right]] = right
        currentLen := right - left + 1
        if currentLen > maxLen {
            maxLen = currentLen
        }
    }

    return maxLen
}

// Minimum Window Substring
func minWindow(s string, t string) string {
    if len(s) < len(t) {
        return ""
    }

    tCount := make(map[byte]int)
    for i := 0; i < len(t); i++ {
        tCount[t[i]]++
    }

    left, right := 0, 0
    minLen := len(s) + 1
    minStart := 0
    required := len(tCount)
    formed := 0
    windowCount := make(map[byte]int)

    for right < len(s) {
        // Add character from right
        c := s[right]
        windowCount[c]++

        if tCount[c] > 0 && windowCount[c] == tCount[c] {
            formed++
        }

        // Try to contract window from left
        for left <= right && formed == required {
            c := s[left]

            // Update minimum window
            if right-left+1 < minLen {
                minLen = right - left + 1
                minStart = left
            }

            // Remove character from left
            windowCount[c]--
            if tCount[c] > 0 && windowCount[c] < tCount[c] {
                formed--
            }

            left++
        }

        right++
    }

    if minLen == len(s)+1 {
        return ""
    }

    return s[minStart : minStart+minLen]
}

// Helper function
func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

// Example usage
func main() {
    // Test Two Sum
    fmt.Println("=== Two Sum ===")
    nums1 := []int{2, 7, 11, 15}
    target1 := 9
    result1 := twoSum(nums1, target1)
    fmt.Printf("Two Sum: %v\n", result1)

    // Test Two Sum Sorted
    fmt.Println("\n=== Two Sum Sorted ===")
    nums2 := []int{2, 7, 11, 15}
    target2 := 9
    result2 := twoSumSorted(nums2, target2)
    fmt.Printf("Two Sum Sorted: %v\n", result2)

    // Test Three Sum
    fmt.Println("\n=== Three Sum ===")
    nums3 := []int{-1, 0, 1, 2, -1, -4}
    result3 := threeSum(nums3)
    fmt.Printf("Three Sum: %v\n", result3)

    // Test Container With Most Water
    fmt.Println("\n=== Container With Most Water ===")
    height1 := []int{1, 8, 6, 2, 5, 4, 8, 3, 7}
    result4 := maxArea(height1)
    fmt.Printf("Max Area: %d\n", result4)

    // Test Remove Duplicates
    fmt.Println("\n=== Remove Duplicates ===")
    nums4 := []int{1, 1, 2, 2, 3, 4, 4, 5}
    length := removeDuplicates(nums4)
    fmt.Printf("Length after removing duplicates: %d\n", length)
    fmt.Printf("Array: %v\n", nums4[:length])

    // Test Move Zeroes
    fmt.Println("\n=== Move Zeroes ===")
    nums5 := []int{0, 1, 0, 3, 12}
    moveZeroes(nums5)
    fmt.Printf("After moving zeroes: %v\n", nums5)

    // Test Valid Palindrome
    fmt.Println("\n=== Valid Palindrome ===")
    s1 := "A man, a plan, a canal: Panama"
    result5 := isPalindrome(s1)
    fmt.Printf("Is '%s' a palindrome? %v\n", s1, result5)

    // Test Trapping Rain Water
    fmt.Println("\n=== Trapping Rain Water ===")
    height2 := []int{0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1}
    result6 := trap(height2)
    fmt.Printf("Trapped water: %d\n", result6)

    // Test Find Anagrams
    fmt.Println("\n=== Find Anagrams ===")
    s2 := "cbaebabacd"
    p := "abc"
    result7 := findAnagrams(s2, p)
    fmt.Printf("Anagram indices in '%s' for '%s': %v\n", s2, p, result7)

    // Test Longest Substring Without Repeating Characters
    fmt.Println("\n=== Longest Substring Without Repeating Characters ===")
    s3 := "abcabcbb"
    result8 := lengthOfLongestSubstring(s3)
    fmt.Printf("Longest substring length in '%s': %d\n", s3, result8)

    // Test Minimum Window Substring
    fmt.Println("\n=== Minimum Window Substring ===")
    s4 := "ADOBECODEBANC"
    t := "ABC"
    result9 := minWindow(s4, t)
    fmt.Printf("Minimum window in '%s' containing '%s': '%s'\n", s4, t, result9)
}
```

## 💻 **Implementation (Node.js)**

```javascript
// Two Sum - Find indices of two numbers that sum to target
function twoSum(nums, target) {
  const numMap = new Map();

  for (let i = 0; i < nums.length; i++) {
    const complement = target - nums[i];
    if (numMap.has(complement)) {
      return [numMap.get(complement), i];
    }
    numMap.set(nums[i], i);
  }

  return []; // No solution found
}

// Two Sum II - Sorted array (Two Pointers approach)
function twoSumSorted(nums, target) {
  let left = 0;
  let right = nums.length - 1;

  while (left < right) {
    const sum = nums[left] + nums[right];

    if (sum === target) {
      return [left, right];
    } else if (sum < target) {
      left++;
    } else {
      right--;
    }
  }

  return []; // No solution found
}

// Three Sum - Find all unique triplets that sum to zero
function threeSum(nums) {
  nums.sort((a, b) => a - b);
  const result = [];

  for (let i = 0; i < nums.length - 2; i++) {
    // Skip duplicates for first number
    if (i > 0 && nums[i] === nums[i - 1]) {
      continue;
    }

    let left = i + 1;
    let right = nums.length - 1;
    const target = -nums[i];

    while (left < right) {
      const sum = nums[left] + nums[right];

      if (sum === target) {
        result.push([nums[i], nums[left], nums[right]]);

        // Skip duplicates for second number
        while (left < right && nums[left] === nums[left + 1]) {
          left++;
        }
        // Skip duplicates for third number
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

// Container With Most Water
function maxArea(height) {
  let left = 0;
  let right = height.length - 1;
  let maxWater = 0;

  while (left < right) {
    // Calculate current area
    const width = right - left;
    const currentHeight = Math.min(height[left], height[right]);
    const currentArea = width * currentHeight;

    if (currentArea > maxWater) {
      maxWater = currentArea;
    }

    // Move pointer with smaller height
    if (height[left] < height[right]) {
      left++;
    } else {
      right--;
    }
  }

  return maxWater;
}

// Remove Duplicates from Sorted Array
function removeDuplicates(nums) {
  if (nums.length === 0) return 0;

  let slow = 0;
  for (let fast = 1; fast < nums.length; fast++) {
    if (nums[fast] !== nums[slow]) {
      slow++;
      nums[slow] = nums[fast];
    }
  }

  return slow + 1;
}

// Move Zeroes to End
function moveZeroes(nums) {
  let slow = 0;

  // Move all non-zero elements to the front
  for (let fast = 0; fast < nums.length; fast++) {
    if (nums[fast] !== 0) {
      [nums[slow], nums[fast]] = [nums[fast], nums[slow]];
      slow++;
    }
  }
}

// Valid Palindrome
function isPalindrome(s) {
  let left = 0;
  let right = s.length - 1;

  while (left < right) {
    // Skip non-alphanumeric characters
    while (left < right && !isAlphanumeric(s[left])) {
      left++;
    }
    while (left < right && !isAlphanumeric(s[right])) {
      right--;
    }

    // Compare characters (case insensitive)
    if (toLowerCase(s[left]) !== toLowerCase(s[right])) {
      return false;
    }

    left++;
    right--;
  }

  return true;
}

function isAlphanumeric(c) {
  return (
    (c >= "a" && c <= "z") || (c >= "A" && c <= "Z") || (c >= "0" && c <= "9")
  );
}

function toLowerCase(c) {
  if (c >= "A" && c <= "Z") {
    return c.toLowerCase();
  }
  return c;
}

// Trapping Rain Water
function trap(height) {
  if (height.length < 3) return 0;

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

// Find All Anagrams in String
function findAnagrams(s, p) {
  if (s.length < p.length) return [];

  const result = [];
  const pCount = new Map();
  const windowCount = new Map();

  // Count characters in pattern
  for (let i = 0; i < p.length; i++) {
    pCount.set(p[i], (pCount.get(p[i]) || 0) + 1);
  }

  // Sliding window
  let left = 0;
  for (let right = 0; right < s.length; right++) {
    // Add character to window
    windowCount.set(s[right], (windowCount.get(s[right]) || 0) + 1);

    // If window size exceeds pattern length, remove left character
    if (right - left + 1 > p.length) {
      windowCount.set(s[left], windowCount.get(s[left]) - 1);
      if (windowCount.get(s[left]) === 0) {
        windowCount.delete(s[left]);
      }
      left++;
    }

    // Check if current window is an anagram
    if (right - left + 1 === p.length && mapsEqual(pCount, windowCount)) {
      result.push(left);
    }
  }

  return result;
}

function mapsEqual(m1, m2) {
  if (m1.size !== m2.size) return false;

  for (const [key, value] of m1) {
    if (m2.get(key) !== value) return false;
  }

  return true;
}

// Longest Substring Without Repeating Characters
function lengthOfLongestSubstring(s) {
  const charMap = new Map();
  let left = 0;
  let maxLen = 0;

  for (let right = 0; right < s.length; right++) {
    if (charMap.has(s[right]) && charMap.get(s[right]) >= left) {
      left = charMap.get(s[right]) + 1;
    }

    charMap.set(s[right], right);
    const currentLen = right - left + 1;
    if (currentLen > maxLen) {
      maxLen = currentLen;
    }
  }

  return maxLen;
}

// Minimum Window Substring
function minWindow(s, t) {
  if (s.length < t.length) return "";

  const tCount = new Map();
  for (let i = 0; i < t.length; i++) {
    tCount.set(t[i], (tCount.get(t[i]) || 0) + 1);
  }

  let left = 0;
  let right = 0;
  let minLen = s.length + 1;
  let minStart = 0;
  const required = tCount.size;
  let formed = 0;
  const windowCount = new Map();

  while (right < s.length) {
    // Add character from right
    const c = s[right];
    windowCount.set(c, (windowCount.get(c) || 0) + 1);

    if (tCount.has(c) && windowCount.get(c) === tCount.get(c)) {
      formed++;
    }

    // Try to contract window from left
    while (left <= right && formed === required) {
      const c = s[left];

      // Update minimum window
      if (right - left + 1 < minLen) {
        minLen = right - left + 1;
        minStart = left;
      }

      // Remove character from left
      windowCount.set(c, windowCount.get(c) - 1);
      if (tCount.has(c) && windowCount.get(c) < tCount.get(c)) {
        formed--;
      }

      left++;
    }

    right++;
  }

  return minLen === s.length + 1
    ? ""
    : s.substring(minStart, minStart + minLen);
}

// Example usage
function main() {
  // Test Two Sum
  console.log("=== Two Sum ===");
  const nums1 = [2, 7, 11, 15];
  const target1 = 9;
  const result1 = twoSum(nums1, target1);
  console.log(`Two Sum: ${JSON.stringify(result1)}`);

  // Test Two Sum Sorted
  console.log("\n=== Two Sum Sorted ===");
  const nums2 = [2, 7, 11, 15];
  const target2 = 9;
  const result2 = twoSumSorted(nums2, target2);
  console.log(`Two Sum Sorted: ${JSON.stringify(result2)}`);

  // Test Three Sum
  console.log("\n=== Three Sum ===");
  const nums3 = [-1, 0, 1, 2, -1, -4];
  const result3 = threeSum(nums3);
  console.log(`Three Sum: ${JSON.stringify(result3)}`);

  // Test Container With Most Water
  console.log("\n=== Container With Most Water ===");
  const height1 = [1, 8, 6, 2, 5, 4, 8, 3, 7];
  const result4 = maxArea(height1);
  console.log(`Max Area: ${result4}`);

  // Test Remove Duplicates
  console.log("\n=== Remove Duplicates ===");
  const nums4 = [1, 1, 2, 2, 3, 4, 4, 5];
  const length = removeDuplicates(nums4);
  console.log(`Length after removing duplicates: ${length}`);
  console.log(`Array: ${JSON.stringify(nums4.slice(0, length))}`);

  // Test Move Zeroes
  console.log("\n=== Move Zeroes ===");
  const nums5 = [0, 1, 0, 3, 12];
  moveZeroes(nums5);
  console.log(`After moving zeroes: ${JSON.stringify(nums5)}`);

  // Test Valid Palindrome
  console.log("\n=== Valid Palindrome ===");
  const s1 = "A man, a plan, a canal: Panama";
  const result5 = isPalindrome(s1);
  console.log(`Is '${s1}' a palindrome? ${result5}`);

  // Test Trapping Rain Water
  console.log("\n=== Trapping Rain Water ===");
  const height2 = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1];
  const result6 = trap(height2);
  console.log(`Trapped water: ${result6}`);

  // Test Find Anagrams
  console.log("\n=== Find Anagrams ===");
  const s2 = "cbaebabacd";
  const p = "abc";
  const result7 = findAnagrams(s2, p);
  console.log(
    `Anagram indices in '${s2}' for '${p}': ${JSON.stringify(result7)}`
  );

  // Test Longest Substring Without Repeating Characters
  console.log("\n=== Longest Substring Without Repeating Characters ===");
  const s3 = "abcabcbb";
  const result8 = lengthOfLongestSubstring(s3);
  console.log(`Longest substring length in '${s3}': ${result8}`);

  // Test Minimum Window Substring
  console.log("\n=== Minimum Window Substring ===");
  const s4 = "ADOBECODEBANC";
  const t = "ABC";
  const result9 = minWindow(s4, t);
  console.log(`Minimum window in '${s4}' containing '${t}': '${result9}'`);
}

main();
```

## ⏱ **Complexity Analysis**

### **Time Complexity**

- **Two Sum**: O(n) - Single pass through array
- **Two Sum Sorted**: O(n) - Two pointers meet in middle
- **Three Sum**: O(n²) - Outer loop + two pointers
- **Container With Most Water**: O(n) - Two pointers traverse once
- **Remove Duplicates**: O(n) - Single pass through array
- **Move Zeroes**: O(n) - Single pass through array
- **Valid Palindrome**: O(n) - Two pointers meet in middle
- **Trapping Rain Water**: O(n) - Two pointers traverse once
- **Find Anagrams**: O(n) - Sliding window approach
- **Longest Substring**: O(n) - Each character visited once
- **Minimum Window**: O(n) - Each character visited at most twice

### **Space Complexity**

- **Most Algorithms**: O(1) - Only using constant extra space
- **Two Sum**: O(n) - Hash map for lookups
- **Find Anagrams**: O(k) - Maps for pattern and window
- **Longest Substring**: O(min(m,n)) - Character map
- **Minimum Window**: O(k) - Maps for pattern and window

## 🚀 **Optimal Solution**

The Two Pointers technique is optimal for many problems because:

1. **Linear Time**: Achieves O(n) time complexity for many problems
2. **Constant Space**: Often uses O(1) extra space
3. **Intuitive**: Mirrors natural problem-solving approach
4. **Versatile**: Works with arrays, strings, linked lists

### **When to Use Two Pointers**

- **Sorted Arrays**: Perfect for finding pairs/triplets
- **Palindromes**: Check from both ends
- **Sliding Window**: Maintain window with two pointers
- **Cycle Detection**: Fast and slow pointers
- **Partitioning**: Separate elements based on condition

## ❓ **Follow-up Questions**

### **How would this scale with X?**

- **Large Arrays**: Two pointers still O(n), very efficient
- **Unsorted Data**: Sort first O(n log n), then apply two pointers
- **Memory Constraints**: Most two pointer solutions use O(1) space

### **How can we optimize further if Y changes?**

- **Duplicate Elements**: Skip duplicates in sorted arrays
- **Multiple Targets**: Use hash map for O(1) lookups
- **Circular Arrays**: Use modulo arithmetic for wraparound

### **What trade-offs exist in different approaches?**

- **Two Pointers vs Hash Map**: Space vs Time complexity
- **Sorting vs Two Pointers**: Preprocessing vs Runtime efficiency
- **Sliding Window vs Brute Force**: O(n) vs O(n²) time complexity
