# Meta Round 1: Coding Interview

## 📋 Interview Format

- **Duration**: 45-60 minutes
- **Format**: Live coding on shared editor (CoderPad, HackerRank)
- **Language**: Any language (Python, Java, C++, JavaScript preferred)
- **Focus**: Problem-solving, algorithm efficiency, and code quality

## 🎯 What Meta Looks For

### Technical Skills

- **Algorithm Mastery**: Deep understanding of data structures and algorithms
- **Code Quality**: Clean, readable, and maintainable code
- **Optimization**: Time and space complexity awareness
- **Problem Decomposition**: Breaking down complex problems into smaller parts

### Soft Skills

- **Communication**: Clear explanation of thought process
- **Collaboration**: Working with interviewer as a coding partner
- **Adaptability**: Handling follow-up questions and modifications

## 🔥 Common Problem Categories

### 1. Arrays and Strings

**Focus**: Meta deals with massive amounts of text and data

#### Example: Two Sum Variations

```python
# Problem: Two Sum with multiple solutions
# Given an array of integers, return indices of two numbers that add up to target

def two_sum_brute_force(nums, target):
    """O(n²) time, O(1) space"""
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []

def two_sum_hash_map(nums, target):
    """O(n) time, O(n) space"""
    num_to_index = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_to_index:
            return [num_to_index[complement], i]
        num_to_index[num] = i
    return []

def two_sum_sorted(nums, target):
    """O(n) time, O(1) space - for sorted array"""
    left, right = 0, len(nums) - 1
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []
```

#### Example: String Pattern Matching

```python
# Problem: Implement strStr() - find first occurrence of needle in haystack
# Meta uses this for search functionality

def str_str_kmp(haystack, needle):
    """O(m+n) time, O(m) space using KMP algorithm"""
    if not needle:
        return 0

    # Build failure function
    failure = [0] * len(needle)
    j = 0
    for i in range(1, len(needle)):
        while j > 0 and needle[i] != needle[j]:
            j = failure[j - 1]
        if needle[i] == needle[j]:
            j += 1
        failure[i] = j

    # Search
    j = 0
    for i in range(len(haystack)):
        while j > 0 and haystack[i] != needle[j]:
            j = failure[j - 1]
        if haystack[i] == needle[j]:
            j += 1
        if j == len(needle):
            return i - j + 1

    return -1

def str_str_rabin_karp(haystack, needle):
    """O(m+n) average time, O(1) space using rolling hash"""
    if not needle:
        return 0

    n, m = len(haystack), len(needle)
    if n < m:
        return -1

    # Rolling hash
    base = 256
    mod = 10**9 + 7

    # Calculate hash of needle
    needle_hash = 0
    for char in needle:
        needle_hash = (needle_hash * base + ord(char)) % mod

    # Calculate hash of first window
    window_hash = 0
    for i in range(m):
        window_hash = (window_hash * base + ord(haystack[i])) % mod

    # Check first window
    if window_hash == needle_hash and haystack[:m] == needle:
        return 0

    # Rolling hash for remaining windows
    base_power = pow(base, m - 1, mod)
    for i in range(1, n - m + 1):
        # Remove first character and add new character
        window_hash = (window_hash - ord(haystack[i-1]) * base_power) % mod
        window_hash = (window_hash * base + ord(haystack[i + m - 1])) % mod

        if window_hash == needle_hash and haystack[i:i+m] == needle:
            return i

    return -1
```

### 2. Dynamic Programming

**Focus**: Meta uses DP for optimization problems in ML and recommendation systems

#### Example: Longest Common Subsequence

```python
# Problem: Find longest common subsequence between two strings
# Used in Meta's content matching and recommendation systems

def lcs_recursive(text1, text2):
    """O(2^(m+n)) time, O(m+n) space - exponential"""
    def helper(i, j):
        if i == len(text1) or j == len(text2):
            return 0

        if text1[i] == text2[j]:
            return 1 + helper(i + 1, j + 1)
        else:
            return max(helper(i + 1, j), helper(i, j + 1))

    return helper(0, 0)

def lcs_memoized(text1, text2):
    """O(m*n) time, O(m*n) space with memoization"""
    memo = {}

    def helper(i, j):
        if (i, j) in memo:
            return memo[(i, j)]

        if i == len(text1) or j == len(text2):
            return 0

        if text1[i] == text2[j]:
            result = 1 + helper(i + 1, j + 1)
        else:
            result = max(helper(i + 1, j), helper(i, j + 1))

        memo[(i, j)] = result
        return result

    return helper(0, 0)

def lcs_dp(text1, text2):
    """O(m*n) time, O(m*n) space with DP table"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]

def lcs_optimized(text1, text2):
    """O(m*n) time, O(min(m,n)) space - space optimized"""
    if len(text1) < len(text2):
        text1, text2 = text2, text1

    m, n = len(text1), len(text2)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                curr[j] = 1 + prev[j-1]
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, prev

    return prev[n]
```

#### Example: Edit Distance

```python
# Problem: Calculate minimum edit distance between two strings
# Used in Meta's spell checker and content similarity

def edit_distance_recursive(word1, word2):
    """O(3^(m+n)) time, O(m+n) space - exponential"""
    def helper(i, j):
        if i == 0:
            return j
        if j == 0:
            return i

        if word1[i-1] == word2[j-1]:
            return helper(i-1, j-1)
        else:
            return 1 + min(
                helper(i-1, j),    # delete
                helper(i, j-1),    # insert
                helper(i-1, j-1)   # replace
            )

    return helper(len(word1), len(word2))

def edit_distance_dp(word1, word2):
    """O(m*n) time, O(m*n) space with DP table"""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # delete
                    dp[i][j-1],    # insert
                    dp[i-1][j-1]   # replace
                )

    return dp[m][n]

def edit_distance_optimized(word1, word2):
    """O(m*n) time, O(min(m,n)) space - space optimized"""
    if len(word1) < len(word2):
        word1, word2 = word2, word1

    m, n = len(word1), len(word2)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
        prev, curr = curr, prev

    return prev[n]
```

### 3. Graphs and Trees

**Focus**: Meta's social network and recommendation systems

#### Example: Friend Suggestions

```python
# Problem: Find mutual friends for friend suggestions
# Given a social graph, find friends of friends who are not already friends

from collections import defaultdict, deque

def find_mutual_friends(graph, user):
    """Find friends of friends who are not already friends"""
    friends = set(graph[user])
    mutual_friends = defaultdict(int)

    for friend in friends:
        for friend_of_friend in graph[friend]:
            if friend_of_friend != user and friend_of_friend not in friends:
                mutual_friends[friend_of_friend] += 1

    # Sort by mutual friend count
    return sorted(mutual_friends.items(), key=lambda x: x[1], reverse=True)

def find_shortest_path(graph, start, end):
    """Find shortest path between two users using BFS"""
    if start == end:
        return [start]

    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        current, path = queue.popleft()

        for neighbor in graph[current]:
            if neighbor == end:
                return path + [neighbor]

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return []  # No path found

def find_connected_components(graph):
    """Find all connected components in the social graph"""
    visited = set()
    components = []

    def dfs(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, component)

    for node in graph:
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)

    return components
```

#### Example: Binary Tree Traversal

```python
# Problem: Various tree traversal methods
# Used in Meta's content organization and search

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def preorder_traversal(root):
    """Root -> Left -> Right"""
    if not root:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)

        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return result

def inorder_traversal(root):
    """Left -> Root -> Right"""
    if not root:
        return []

    result = []
    stack = []
    current = root

    while stack or current:
        while current:
            stack.append(current)
            current = current.left

        current = stack.pop()
        result.append(current.val)
        current = current.right

    return result

def postorder_traversal(root):
    """Left -> Right -> Root"""
    if not root:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)

        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)

    return result[::-1]

def level_order_traversal(root):
    """Level by level traversal using BFS"""
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result
```

### 4. Sliding Window

**Focus**: Meta uses sliding window for real-time analytics and monitoring

#### Example: Maximum Sum Subarray

```python
# Problem: Find maximum sum of contiguous subarray
# Used in Meta's ad revenue optimization

def max_subarray_brute_force(nums):
    """O(n²) time, O(1) space"""
    max_sum = float('-inf')

    for i in range(len(nums)):
        current_sum = 0
        for j in range(i, len(nums)):
            current_sum += nums[j]
            max_sum = max(max_sum, current_sum)

    return max_sum

def max_subarray_kadane(nums):
    """O(n) time, O(1) space - Kadane's algorithm"""
    max_sum = current_sum = nums[0]

    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)

    return max_sum

def max_subarray_sliding_window(nums, k):
    """O(n) time, O(1) space - sliding window for fixed size"""
    if len(nums) < k:
        return 0

    window_sum = sum(nums[:k])
    max_sum = window_sum

    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i-k] + nums[i]
        max_sum = max(max_sum, window_sum)

    return max_sum
```

#### Example: Longest Substring Without Repeating Characters

```python
# Problem: Find longest substring without repeating characters
# Used in Meta's content analysis and text processing

def length_of_longest_substring_brute_force(s):
    """O(n³) time, O(min(m,n)) space"""
    max_length = 0

    for i in range(len(s)):
        for j in range(i, len(s)):
            if len(set(s[i:j+1])) == j - i + 1:
                max_length = max(max_length, j - i + 1)

    return max_length

def length_of_longest_substring_sliding_window(s):
    """O(n) time, O(min(m,n)) space - sliding window"""
    char_map = {}
    left = max_length = 0

    for right in range(len(s)):
        if s[right] in char_map and char_map[s[right]] >= left:
            left = char_map[s[right]] + 1

        char_map[s[right]] = right
        max_length = max(max_length, right - left + 1)

    return max_length

def length_of_longest_substring_set(s):
    """O(n) time, O(min(m,n)) space - using set"""
    char_set = set()
    left = max_length = 0

    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1

        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)

    return max_length
```

### 5. Heap and Priority Queue

**Focus**: Meta uses heaps for recommendation systems and task scheduling

#### Example: Top K Elements

```python
# Problem: Find top K frequent elements
# Used in Meta's trending topics and recommendation systems

import heapq
from collections import Counter

def top_k_frequent_heap(nums, k):
    """O(n log k) time, O(n) space using min heap"""
    count = Counter(nums)

    # Use min heap of size k
    heap = []
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)

    return [num for freq, num in heap]

def top_k_frequent_quickselect(nums, k):
    """O(n) average time, O(n) space using quickselect"""
    count = Counter(nums)
    unique = list(count.keys())

    def partition(left, right, pivot_index):
        pivot_freq = count[unique[pivot_index]]
        unique[pivot_index], unique[right] = unique[right], unique[pivot_index]

        store_index = left
        for i in range(left, right):
            if count[unique[i]] < pivot_freq:
                unique[store_index], unique[i] = unique[i], unique[store_index]
                store_index += 1

        unique[right], unique[store_index] = unique[store_index], unique[right]
        return store_index

    def quickselect(left, right, k_smallest):
        if left == right:
            return

        pivot_index = partition(left, right, right)

        if k_smallest == pivot_index:
            return
        elif k_smallest < pivot_index:
            quickselect(left, pivot_index - 1, k_smallest)
        else:
            quickselect(pivot_index + 1, right, k_smallest)

    n = len(unique)
    quickselect(0, n - 1, n - k)
    return unique[n - k:]

def top_k_frequent_bucket_sort(nums, k):
    """O(n) time, O(n) space using bucket sort"""
    count = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]

    for num, freq in count.items():
        buckets[freq].append(num)

    result = []
    for i in range(len(buckets) - 1, 0, -1):
        result.extend(buckets[i])
        if len(result) >= k:
            break

    return result[:k]
```

## 🧪 Testing Patterns

### Unit Testing

```python
import unittest

class TestTwoSum(unittest.TestCase):
    def test_two_sum_brute_force(self):
        self.assertEqual(two_sum_brute_force([2, 7, 11, 15], 9), [0, 1])
        self.assertEqual(two_sum_brute_force([3, 2, 4], 6), [1, 2])
        self.assertEqual(two_sum_brute_force([3, 3], 6), [0, 1])
        self.assertEqual(two_sum_brute_force([1, 2, 3], 7), [])

    def test_two_sum_hash_map(self):
        self.assertEqual(two_sum_hash_map([2, 7, 11, 15], 9), [0, 1])
        self.assertEqual(two_sum_hash_map([3, 2, 4], 6), [1, 2])
        self.assertEqual(two_sum_hash_map([3, 3], 6), [0, 1])
        self.assertEqual(two_sum_hash_map([1, 2, 3], 7), [])

if __name__ == '__main__':
    unittest.main()
```

### Performance Testing

```python
import time
import random

def performance_test():
    # Generate test data
    nums = [random.randint(-1000, 1000) for _ in range(1000)]
    target = random.randint(-2000, 2000)

    # Test brute force
    start = time.time()
    result1 = two_sum_brute_force(nums, target)
    time1 = time.time() - start

    # Test hash map
    start = time.time()
    result2 = two_sum_hash_map(nums, target)
    time2 = time.time() - start

    print(f"Brute force: {time1:.6f}s")
    print(f"Hash map: {time2:.6f}s")
    print(f"Speedup: {time1/time2:.2f}x")

if __name__ == '__main__':
    performance_test()
```

## 🎯 Interview Tips

### Before the Interview

1. **Practice LeetCode**: Focus on medium and hard problems
2. **Review Fundamentals**: Arrays, strings, trees, graphs, DP
3. **Time Yourself**: Practice solving problems in 20-30 minutes
4. **Study Meta Problems**: Look at Meta-specific coding questions

### During the Interview

1. **Clarify Requirements**: Ask about edge cases and constraints
2. **Think Out Loud**: Explain your approach before coding
3. **Start Simple**: Get brute force solution working first
4. **Optimize Step by Step**: Improve time/space complexity
5. **Test Your Code**: Walk through examples

### Common Pitfalls to Avoid

1. **Silent Coding**: Always explain your thought process
2. **Premature Optimization**: Get it working first
3. **Ignoring Edge Cases**: Ask about null inputs, empty arrays
4. **Poor Variable Names**: Use descriptive names
5. **Not Testing**: Always test with examples

## 📚 Preparation Resources

### Coding Resources

- [LeetCode](https://leetcode.com/) - Focus on Meta problems
- [HackerRank](https://www.hackerrank.com/) - Algorithm practice
- [DSA Questions](../shared/dsa-questions.md) - Comprehensive guide

### Meta-Specific

- [Meta Interview Guide](https://www.metacareers.com/interview-prep/)
- [Meta Coding Problems](https://leetcode.com/company/meta/)
- [Meta Engineering Blog](https://engineering.fb.com/)

## 🔗 Related Content

- [System Design Round 2](round2-system-design.md) - For architecture discussions
- [Product Sense Round 3](round3-product-sense.md) - For product thinking
- [Behavioral Round 4](round4-behavioral.md) - For behavioral questions
