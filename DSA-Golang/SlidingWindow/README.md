# Sliding Window Pattern

> **Master sliding window techniques for array and string problems with Go implementations**

## 📋 Problems

### **Fixed Window Size**
- [Maximum Sum Subarray of Size K](./MaximumSumSubarrayOfSizeK.md) - Find maximum sum of fixed window
- [Average of Subarrays of Size K](./AverageOfSubarraysOfSizeK.md) - Calculate averages
- [First Negative Number in Every Window](./FirstNegativeNumberInEveryWindow.md) - Find first negative in each window
- [Count Anagrams](./CountAnagrams.md) - Count anagrams in string
- [Maximum of All Subarrays of Size K](./MaximumOfAllSubarraysOfSizeK.md) - Find maximum in each window

### **Variable Window Size**
- [Longest Substring Without Repeating Characters](./LongestSubstringWithoutRepeatingCharacters.md) - Find longest unique substring
- [Minimum Window Substring](./MinimumWindowSubstring.md) - Find minimum window containing all characters
- [Longest Substring with At Most K Distinct Characters](./LongestSubstringWithAtMostKDistinctCharacters.md) - Find longest substring with K distinct chars
- [Longest Repeating Character Replacement](./LongestRepeatingCharacterReplacement.md) - Replace characters to get longest substring
- [Fruit Into Baskets](./FruitIntoBaskets.md) - Collect maximum fruits

### **Advanced Sliding Window**
- [Subarray Sum Equals K](./SubarraySumEqualsK.md) - Find subarrays with sum K
- [Permutation in String](./PermutationInString.md) - Check if permutation exists
- [Find All Anagrams in a String](./FindAllAnagramsInString.md) - Find all anagram occurrences
- [Sliding Window Maximum](./SlidingWindowMaximum.md) - Find maximum in sliding window
- [Longest Subarray with Sum at Most K](./LongestSubarrayWithSumAtMostK.md) - Find longest subarray with sum ≤ K

---

## 🎯 Key Concepts

### **Sliding Window Types**
1. **Fixed Window**: Window size remains constant
2. **Variable Window**: Window size changes based on condition
3. **Two Pointers**: Use left and right pointers to define window

### **When to Use Sliding Window**
- **Subarray/Substring Problems**: Find optimal subarray/substring
- **Contiguous Elements**: Work with consecutive elements
- **Optimization Problems**: Find maximum/minimum in window
- **Frequency Problems**: Count characters/elements in window

### **Common Patterns**
- **Expand Right**: Increase window size
- **Contract Left**: Decrease window size
- **Maintain Invariant**: Keep window valid
- **Update Result**: Track optimal solution

---

## 🛠️ Go-Specific Tips

### **Fixed Window Template**
```go
func maxSumSubarray(nums []int, k int) int {
    if len(nums) < k {
        return 0
    }
    
    // Calculate sum of first window
    windowSum := 0
    for i := 0; i < k; i++ {
        windowSum += nums[i]
    }
    
    maxSum := windowSum
    
    // Slide the window
    for i := k; i < len(nums); i++ {
        windowSum = windowSum - nums[i-k] + nums[i]
        maxSum = max(maxSum, windowSum)
    }
    
    return maxSum
}
```

### **Variable Window Template**
```go
func longestSubstring(s string, k int) int {
    left := 0
    maxLen := 0
    charCount := make(map[byte]int)
    
    for right := 0; right < len(s); right++ {
        // Expand window
        charCount[s[right]]++
        
        // Contract window while maintaining condition
        for len(charCount) > k {
            charCount[s[left]]--
            if charCount[s[left]] == 0 {
                delete(charCount, s[left])
            }
            left++
        }
        
        // Update result
        maxLen = max(maxLen, right-left+1)
    }
    
    return maxLen
}
```

### **Two Pointers Technique**
```go
func twoSum(nums []int, target int) []int {
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
    
    return nil
}
```

### **Frequency Map with Sliding Window**
```go
func minWindow(s string, t string) string {
    if len(s) < len(t) {
        return ""
    }
    
    // Count characters in target string
    targetCount := make(map[byte]int)
    for i := 0; i < len(t); i++ {
        targetCount[t[i]]++
    }
    
    left := 0
    minLen := len(s) + 1
    minStart := 0
    matched := 0
    
    for right := 0; right < len(s); right++ {
        // Expand window
        if targetCount[s[right]] > 0 {
            matched++
        }
        targetCount[s[right]]--
        
        // Contract window
        for matched == len(t) {
            if right-left+1 < minLen {
                minLen = right - left + 1
                minStart = left
            }
            
            targetCount[s[left]]++
            if targetCount[s[left]] > 0 {
                matched--
            }
            left++
        }
    }
    
    if minLen > len(s) {
        return ""
    }
    
    return s[minStart : minStart+minLen]
}
```

---

## 🎯 Interview Tips

### **How to Identify Sliding Window Problems**
1. **Subarray/Substring**: Find optimal contiguous elements
2. **Window Size**: Fixed or variable window size
3. **Optimization**: Find maximum/minimum in window
4. **Frequency**: Count elements in window

### **Common Sliding Window Problem Patterns**
- **Maximum Sum**: Find maximum sum in window
- **Longest Substring**: Find longest substring with condition
- **Minimum Window**: Find minimum window with condition
- **Anagram Detection**: Find anagrams in string
- **Two Pointers**: Use pointers to define window

### **Optimization Tips**
- **Pre-calculate**: Calculate first window separately
- **Incremental Updates**: Update window incrementally
- **Early Termination**: Stop when condition is met
- **Space Optimization**: Use arrays instead of maps when possible
