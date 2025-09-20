# Data Structures and Algorithms Questions

## 📚 Table of Contents

1. [Arrays and Strings](#arrays-and-strings)
2. [Linked Lists](#linked-lists)
3. [Trees and Graphs](#trees-and-graphs)
4. [Dynamic Programming](#dynamic-programming)
5. [Greedy Algorithms](#greedy-algorithms)
6. [Backtracking](#backtracking)
7. [Sliding Window](#sliding-window)
8. [Two Pointers](#two-pointers)
9. [Heap and Priority Queue](#heap-and-priority-queue)
10. [Trie](#trie)
11. [Union Find](#union-find)
12. [Bit Manipulation](#bit-manipulation)

## Arrays and Strings

### Easy Problems

#### 1. Two Sum

**Problem**: Given an array of integers, return indices of two numbers that add up to target.

```python
def two_sum(nums, target):
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []
```

**Time Complexity**: O(n)  
**Space Complexity**: O(n)

#### 2. Valid Parentheses

**Problem**: Check if string has valid parentheses.

```python
def is_valid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)

    return not stack
```

**Time Complexity**: O(n)  
**Space Complexity**: O(n)

#### 3. Maximum Subarray

**Problem**: Find maximum sum of contiguous subarray.

```python
def max_subarray(nums):
    max_sum = current_sum = nums[0]

    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)

    return max_sum
```

**Time Complexity**: O(n)  
**Space Complexity**: O(1)

### Medium Problems

#### 4. Longest Substring Without Repeating Characters

**Problem**: Find length of longest substring without repeating characters.

```python
def length_of_longest_substring(s):
    char_map = {}
    left = max_length = 0

    for right in range(len(s)):
        if s[right] in char_map and char_map[s[right]] >= left:
            left = char_map[s[right]] + 1

        char_map[s[right]] = right
        max_length = max(max_length, right - left + 1)

    return max_length
```

**Time Complexity**: O(n)  
**Space Complexity**: O(min(m,n))

#### 5. Group Anagrams

**Problem**: Group strings that are anagrams of each other.

```python
def group_anagrams(strs):
    anagram_map = {}

    for s in strs:
        key = ''.join(sorted(s))
        if key not in anagram_map:
            anagram_map[key] = []
        anagram_map[key].append(s)

    return list(anagram_map.values())
```

**Time Complexity**: O(n _ m log m)  
**Space Complexity**: O(n _ m)

#### 6. Product of Array Except Self

**Problem**: Return array where each element is product of all other elements.

```python
def product_except_self(nums):
    n = len(nums)
    result = [1] * n

    # Left pass
    for i in range(1, n):
        result[i] = result[i-1] * nums[i-1]

    # Right pass
    right_product = 1
    for i in range(n-1, -1, -1):
        result[i] *= right_product
        right_product *= nums[i]

    return result
```

**Time Complexity**: O(n)  
**Space Complexity**: O(1)

### Hard Problems

#### 7. Longest Consecutive Sequence

**Problem**: Find length of longest consecutive sequence.

```python
def longest_consecutive(nums):
    num_set = set(nums)
    max_length = 0

    for num in num_set:
        if num - 1 not in num_set:
            current_length = 1
            while num + 1 in num_set:
                num += 1
                current_length += 1
            max_length = max(max_length, current_length)

    return max_length
```

**Time Complexity**: O(n)  
**Space Complexity**: O(n)

#### 8. Trapping Rain Water

**Problem**: Calculate trapped rainwater.

```python
def trap(height):
    if not height:
        return 0

    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0

    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1

    return water
```

**Time Complexity**: O(n)  
**Space Complexity**: O(1)

## Linked Lists

### Easy Problems

#### 9. Reverse Linked List

**Problem**: Reverse a singly linked list.

```python
def reverse_list(head):
    prev = None
    current = head

    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp

    return prev
```

**Time Complexity**: O(n)  
**Space Complexity**: O(1)

#### 10. Merge Two Sorted Lists

**Problem**: Merge two sorted linked lists.

```python
def merge_two_lists(l1, l2):
    dummy = ListNode(0)
    current = dummy

    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next

    current.next = l1 or l2
    return dummy.next
```

**Time Complexity**: O(n + m)  
**Space Complexity**: O(1)

### Medium Problems

#### 11. Remove Nth Node From End

**Problem**: Remove nth node from end of list.

```python
def remove_nth_from_end(head, n):
    dummy = ListNode(0)
    dummy.next = head
    first = second = dummy

    for _ in range(n + 1):
        first = first.next

    while first:
        first = first.next
        second = second.next

    second.next = second.next.next
    return dummy.next
```

**Time Complexity**: O(n)  
**Space Complexity**: O(1)

#### 12. Copy List with Random Pointer

**Problem**: Deep copy linked list with random pointers.

```python
def copy_random_list(head):
    if not head:
        return None

    # Create mapping
    old_to_new = {}
    current = head

    while current:
        old_to_new[current] = Node(current.val)
        current = current.next

    # Set next and random pointers
    current = head
    while current:
        if current.next:
            old_to_new[current].next = old_to_new[current.next]
        if current.random:
            old_to_new[current].random = old_to_new[current.random]
        current = current.next

    return old_to_new[head]
```

**Time Complexity**: O(n)  
**Space Complexity**: O(n)

## Trees and Graphs

### Easy Problems

#### 13. Maximum Depth of Binary Tree

**Problem**: Find maximum depth of binary tree.

```python
def max_depth(root):
    if not root:
        return 0

    return 1 + max(max_depth(root.left), max_depth(root.right))
```

**Time Complexity**: O(n)  
**Space Complexity**: O(h)

#### 14. Same Tree

**Problem**: Check if two binary trees are identical.

```python
def is_same_tree(p, q):
    if not p and not q:
        return True
    if not p or not q:
        return False

    return (p.val == q.val and
            is_same_tree(p.left, q.left) and
            is_same_tree(p.right, q.right))
```

**Time Complexity**: O(n)  
**Space Complexity**: O(h)

### Medium Problems

#### 15. Binary Tree Level Order Traversal

**Problem**: Return level order traversal of binary tree.

```python
def level_order(root):
    if not root:
        return []

    result = []
    queue = [root]

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.pop(0)
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result
```

**Time Complexity**: O(n)  
**Space Complexity**: O(w)

#### 16. Validate Binary Search Tree

**Problem**: Check if binary tree is valid BST.

```python
def is_valid_bst(root):
    def validate(node, min_val, max_val):
        if not node:
            return True

        if node.val <= min_val or node.val >= max_val:
            return False

        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))

    return validate(root, float('-inf'), float('inf'))
```

**Time Complexity**: O(n)  
**Space Complexity**: O(h)

### Hard Problems

#### 17. Serialize and Deserialize Binary Tree

**Problem**: Serialize and deserialize binary tree.

```python
def serialize(root):
    def preorder(node):
        if not node:
            vals.append('null')
        else:
            vals.append(str(node.val))
            preorder(node.left)
            preorder(node.right)

    vals = []
    preorder(root)
    return ','.join(vals)

def deserialize(data):
    def build():
        val = next(vals)
        if val == 'null':
            return None

        node = TreeNode(int(val))
        node.left = build()
        node.right = build()
        return node

    vals = iter(data.split(','))
    return build()
```

**Time Complexity**: O(n)  
**Space Complexity**: O(n)

## Dynamic Programming

### Easy Problems

#### 18. Climbing Stairs

**Problem**: Find number of ways to climb n stairs.

```python
def climb_stairs(n):
    if n <= 2:
        return n

    prev2, prev1 = 1, 2

    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current

    return prev1
```

**Time Complexity**: O(n)  
**Space Complexity**: O(1)

#### 19. House Robber

**Problem**: Find maximum money that can be robbed.

```python
def rob(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    prev2, prev1 = nums[0], max(nums[0], nums[1])

    for i in range(2, len(nums)):
        current = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, current

    return prev1
```

**Time Complexity**: O(n)  
**Space Complexity**: O(1)

### Medium Problems

#### 20. Longest Increasing Subsequence

**Problem**: Find length of longest increasing subsequence.

```python
def length_of_lis(nums):
    if not nums:
        return 0

    dp = [1] * len(nums)

    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)
```

**Time Complexity**: O(n²)  
**Space Complexity**: O(n)

#### 21. Word Break

**Problem**: Check if string can be segmented into dictionary words.

```python
def word_break(s, word_dict):
    word_set = set(word_dict)
    dp = [False] * (len(s) + 1)
    dp[0] = True

    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    return dp[len(s)]
```

**Time Complexity**: O(n²)  
**Space Complexity**: O(n)

### Hard Problems

#### 22. Edit Distance

**Problem**: Find minimum edit distance between two strings.

```python
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]
```

**Time Complexity**: O(m _ n)  
**Space Complexity**: O(m _ n)

## Greedy Algorithms

### Easy Problems

#### 23. Best Time to Buy and Sell Stock

**Problem**: Find maximum profit from buying and selling stock.

```python
def max_profit(prices):
    if not prices:
        return 0

    min_price = prices[0]
    max_profit = 0

    for price in prices[1:]:
        max_profit = max(max_profit, price - min_price)
        min_price = min(min_price, price)

    return max_profit
```

**Time Complexity**: O(n)  
**Space Complexity**: O(1)

#### 24. Jump Game

**Problem**: Check if can reach last index.

```python
def can_jump(nums):
    max_reach = 0

    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])

    return True
```

**Time Complexity**: O(n)  
**Space Complexity**: O(1)

### Medium Problems

#### 25. Gas Station

**Problem**: Find starting gas station to complete circuit.

```python
def can_complete_circuit(gas, cost):
    total_tank = current_tank = start = 0

    for i in range(len(gas)):
        total_tank += gas[i] - cost[i]
        current_tank += gas[i] - cost[i]

        if current_tank < 0:
            start = i + 1
            current_tank = 0

    return start if total_tank >= 0 else -1
```

**Time Complexity**: O(n)  
**Space Complexity**: O(1)

## Backtracking

### Medium Problems

#### 26. Generate Parentheses

**Problem**: Generate all valid parentheses combinations.

```python
def generate_parenthesis(n):
    result = []

    def backtrack(current, open_count, close_count):
        if len(current) == 2 * n:
            result.append(current)
            return

        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)

        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)

    backtrack('', 0, 0)
    return result
```

**Time Complexity**: O(4^n / √n)  
**Space Complexity**: O(4^n / √n)

#### 27. Subsets

**Problem**: Generate all possible subsets.

```python
def subsets(nums):
    result = []

    def backtrack(start, current):
        result.append(current[:])

        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result
```

**Time Complexity**: O(2^n)  
**Space Complexity**: O(2^n)

### Hard Problems

#### 28. N-Queens

**Problem**: Place n queens on n×n chessboard.

```python
def solve_n_queens(n):
    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]

    def is_safe(row, col):
        for i in range(row):
            if board[i][col] == 'Q':
                return False

        for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
            if board[i][j] == 'Q':
                return False

        for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
            if board[i][j] == 'Q':
                return False

        return True

    def backtrack(row):
        if row == n:
            result.append([''.join(row) for row in board])
            return

        for col in range(n):
            if is_safe(row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'

    backtrack(0)
    return result
```

**Time Complexity**: O(n!)  
**Space Complexity**: O(n²)

## Sliding Window

### Medium Problems

#### 29. Longest Substring with At Most K Distinct Characters

**Problem**: Find longest substring with at most k distinct characters.

```python
def length_of_longest_substring_k_distinct(s, k):
    if k == 0:
        return 0

    char_count = {}
    left = max_length = 0

    for right in range(len(s)):
        char_count[s[right]] = char_count.get(s[right], 0) + 1

        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1

        max_length = max(max_length, right - left + 1)

    return max_length
```

**Time Complexity**: O(n)  
**Space Complexity**: O(k)

#### 30. Minimum Window Substring

**Problem**: Find minimum window in s that contains all characters in t.

```python
def min_window(s, t):
    if not s or not t:
        return ""

    dict_t = {}
    for char in t:
        dict_t[char] = dict_t.get(char, 0) + 1

    required = len(dict_t)
    formed = 0
    window_counts = {}

    left = right = 0
    ans = float('inf'), None, None

    while right < len(s):
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1

        if char in dict_t and window_counts[char] == dict_t[char]:
            formed += 1

        while left <= right and formed == required:
            char = s[left]

            if right - left + 1 < ans[0]:
                ans = (right - left + 1, left, right)

            window_counts[char] -= 1
            if char in dict_t and window_counts[char] < dict_t[char]:
                formed -= 1

            left += 1

        right += 1

    return "" if ans[0] == float('inf') else s[ans[1]:ans[2] + 1]
```

**Time Complexity**: O(|s| + |t|)  
**Space Complexity**: O(|s| + |t|)

## Two Pointers

### Easy Problems

#### 31. Valid Palindrome

**Problem**: Check if string is palindrome.

```python
def is_palindrome(s):
    left, right = 0, len(s) - 1

    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1

        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True
```

**Time Complexity**: O(n)  
**Space Complexity**: O(1)

#### 32. Two Sum II - Input Array Is Sorted

**Problem**: Find two numbers that add up to target in sorted array.

```python
def two_sum_sorted(numbers, target):
    left, right = 0, len(numbers) - 1

    while left < right:
        current_sum = numbers[left] + numbers[right]
        if current_sum == target:
            return [left + 1, right + 1]
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return []
```

**Time Complexity**: O(n)  
**Space Complexity**: O(1)

### Medium Problems

#### 33. Container With Most Water

**Problem**: Find container that holds most water.

```python
def max_area(height):
    left, right = 0, len(height) - 1
    max_area = 0

    while left < right:
        width = right - left
        current_area = width * min(height[left], height[right])
        max_area = max(max_area, current_area)

        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_area
```

**Time Complexity**: O(n)  
**Space Complexity**: O(1)

## Heap and Priority Queue

### Medium Problems

#### 34. Kth Largest Element in Array

**Problem**: Find kth largest element in unsorted array.

```python
import heapq

def find_kth_largest(nums, k):
    return heapq.nlargest(k, nums)[-1]

def find_kth_largest_heap(nums, k):
    heap = nums[:k]
    heapq.heapify(heap)

    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)

    return heap[0]
```

**Time Complexity**: O(n log k)  
**Space Complexity**: O(k)

#### 35. Merge k Sorted Lists

**Problem**: Merge k sorted linked lists.

```python
def merge_k_lists(lists):
    if not lists:
        return None

    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))

    dummy = ListNode(0)
    current = dummy

    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = node
        current = current.next

        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
```

**Time Complexity**: O(n log k)  
**Space Complexity**: O(k)

## Trie

### Medium Problems

#### 36. Implement Trie

**Problem**: Implement trie data structure.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

**Time Complexity**: O(m) for insert/search/starts_with  
**Space Complexity**: O(ALPHABET_SIZE _ N _ M)

## Union Find

### Medium Problems

#### 37. Number of Islands

**Problem**: Count number of islands in 2D grid.

```python
def num_islands(grid):
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    islands = 0

    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            grid[r][c] != '1'):
            return

        grid[r][c] = '0'  # Mark as visited
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                islands += 1
                dfs(r, c)

    return islands
```

**Time Complexity**: O(m _ n)  
**Space Complexity**: O(m _ n)

## Bit Manipulation

### Easy Problems

#### 38. Single Number

**Problem**: Find single number that appears once.

```python
def single_number(nums):
    result = 0
    for num in nums:
        result ^= num
    return result
```

**Time Complexity**: O(n)  
**Space Complexity**: O(1)

#### 39. Missing Number

**Problem**: Find missing number in array containing n distinct numbers.

```python
def missing_number(nums):
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum
```

**Time Complexity**: O(n)  
**Space Complexity**: O(1)

### Medium Problems

#### 40. Counting Bits

**Problem**: Count number of 1 bits for each number from 0 to n.

```python
def count_bits(n):
    result = [0] * (n + 1)

    for i in range(1, n + 1):
        result[i] = result[i >> 1] + (i & 1)

    return result
```

**Time Complexity**: O(n)  
**Space Complexity**: O(n)

## 🎯 Interview Tips

### Problem-Solving Strategy

1. **Understand the Problem**: Read carefully and ask clarifying questions
2. **Think of Examples**: Work through small examples manually
3. **Identify Patterns**: Look for common algorithmic patterns
4. **Start Simple**: Begin with brute force, then optimize
5. **Test Your Solution**: Walk through examples and edge cases

### Common Patterns

- **Two Pointers**: For sorted arrays and palindromes
- **Sliding Window**: For substring problems
- **Hash Map**: For frequency counting and lookups
- **Stack**: For matching and parsing problems
- **Queue**: For BFS and level-order traversal
- **Heap**: For top-k and priority problems
- **Trie**: For string prefix problems
- **Union Find**: For connectivity problems

### Time and Space Complexity

- **Time Complexity**: Always consider the worst case
- **Space Complexity**: Include auxiliary space used
- **Trade-offs**: Sometimes space for time or vice versa
- **Optimization**: Look for ways to improve both

This comprehensive guide covers the most important DSA problems commonly asked in technical interviews, with solutions in Python and complexity analysis for each problem.
