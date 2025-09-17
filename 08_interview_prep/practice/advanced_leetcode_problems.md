# Advanced LeetCode Problems

Comprehensive collection of advanced LeetCode problems for senior engineering interviews.

## 🎯 Hard Difficulty Problems

### Problem 1: Serialize and Deserialize Binary Tree
**Difficulty**: Hard  
**Time Complexity**: O(n)  
**Space Complexity**: O(n)

```python
# Serialize and Deserialize Binary Tree
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string."""
        if not root:
            return "null"
        
        def dfs(node):
            if not node:
                return "null"
            return str(node.val) + "," + dfs(node.left) + "," + dfs(node.right)
        
        return dfs(root)
    
    def deserialize(self, data):
        """Decodes your encoded data to tree."""
        if data == "null":
            return None
        
        values = data.split(",")
        self.index = 0
        
        def dfs():
            if self.index >= len(values) or values[self.index] == "null":
                self.index += 1
                return None
            
            node = TreeNode(int(values[self.index]))
            self.index += 1
            node.left = dfs()
            node.right = dfs()
            return node
        
        return dfs()

# Alternative approach using BFS
class CodecBFS:
    def serialize(self, root):
        """Encodes a tree to a single string using BFS."""
        if not root:
            return "null"
        
        queue = [root]
        result = []
        
        while queue:
            node = queue.pop(0)
            if node:
                result.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append("null")
        
        return ",".join(result)
    
    def deserialize(self, data):
        """Decodes your encoded data to tree using BFS."""
        if data == "null":
            return None
        
        values = data.split(",")
        root = TreeNode(int(values[0]))
        queue = [root]
        i = 1
        
        while queue and i < len(values):
            node = queue.pop(0)
            
            if i < len(values) and values[i] != "null":
                node.left = TreeNode(int(values[i]))
                queue.append(node.left)
            i += 1
            
            if i < len(values) and values[i] != "null":
                node.right = TreeNode(int(values[i]))
                queue.append(node.right)
            i += 1
        
        return root
```

### Problem 2: Merge k Sorted Lists
**Difficulty**: Hard  
**Time Complexity**: O(n log k)  
**Space Complexity**: O(1)

```python
# Merge k Sorted Lists
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeKLists(self, lists):
        """Merge k sorted linked lists and return it as one sorted list."""
        if not lists:
            return None
        
        # Approach 1: Divide and Conquer
        def mergeTwoLists(l1, l2):
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
        
        def mergeKListsHelper(lists, start, end):
            if start == end:
                return lists[start]
            if start + 1 == end:
                return mergeTwoLists(lists[start], lists[end])
            
            mid = (start + end) // 2
            left = mergeKListsHelper(lists, start, mid)
            right = mergeKListsHelper(lists, mid + 1, end)
            return mergeTwoLists(left, right)
        
        return mergeKListsHelper(lists, 0, len(lists) - 1)
    
    def mergeKListsHeap(self, lists):
        """Merge k sorted linked lists using min heap."""
        import heapq
        
        if not lists:
            return None
        
        # Create min heap
        heap = []
        dummy = ListNode(0)
        current = dummy
        
        # Add first node of each list to heap
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(heap, (lst.val, i, lst))
        
        while heap:
            val, i, node = heapq.heappop(heap)
            current.next = node
            current = current.next
            
            if node.next:
                heapq.heappush(heap, (node.next.val, i, node.next))
        
        return dummy.next
```

### Problem 3: Longest Valid Parentheses
**Difficulty**: Hard  
**Time Complexity**: O(n)  
**Space Complexity**: O(n)

```python
# Longest Valid Parentheses
class Solution:
    def longestValidParentheses(self, s):
        """Find the length of the longest valid parentheses substring."""
        if not s:
            return 0
        
        # Approach 1: Using Stack
        def usingStack():
            stack = [-1]  # Initialize with -1 to handle edge cases
            max_len = 0
            
            for i, char in enumerate(s):
                if char == '(':
                    stack.append(i)
                else:  # char == ')'
                    stack.pop()
                    if not stack:
                        stack.append(i)  # New starting point
                    else:
                        max_len = max(max_len, i - stack[-1])
            
            return max_len
        
        # Approach 2: Using Dynamic Programming
        def usingDP():
            n = len(s)
            dp = [0] * n
            max_len = 0
            
            for i in range(1, n):
                if s[i] == ')':
                    if s[i-1] == '(':
                        dp[i] = (dp[i-2] if i >= 2 else 0) + 2
                    elif i - dp[i-1] > 0 and s[i - dp[i-1] - 1] == '(':
                        dp[i] = dp[i-1] + (dp[i - dp[i-1] - 2] if i - dp[i-1] >= 2 else 0) + 2
                    max_len = max(max_len, dp[i])
            
            return max_len
        
        # Approach 3: Using Two Pointers
        def usingTwoPointers():
            left = right = 0
            max_len = 0
            
            # Left to right pass
            for char in s:
                if char == '(':
                    left += 1
                else:
                    right += 1
                
                if left == right:
                    max_len = max(max_len, 2 * right)
                elif right > left:
                    left = right = 0
            
            # Right to left pass
            left = right = 0
            for char in reversed(s):
                if char == '(':
                    left += 1
                else:
                    right += 1
                
                if left == right:
                    max_len = max(max_len, 2 * left)
                elif left > right:
                    left = right = 0
            
            return max_len
        
        return usingStack()
```

## 🚀 Dynamic Programming Problems

### Problem 4: Edit Distance
**Difficulty**: Hard  
**Time Complexity**: O(m * n)  
**Space Complexity**: O(m * n)

```python
# Edit Distance (Levenshtein Distance)
class Solution:
    def minDistance(self, word1, word2):
        """Find the minimum number of operations to convert word1 to word2."""
        m, n = len(word1), len(word2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # Delete
                        dp[i][j-1],    # Insert
                        dp[i-1][j-1]   # Replace
                    )
        
        return dp[m][n]
    
    def minDistanceOptimized(self, word1, word2):
        """Optimized space complexity version."""
        m, n = len(word1), len(word2)
        
        # Use only two rows
        prev = [j for j in range(n + 1)]
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

### Problem 5: Maximum Subarray Sum
**Difficulty**: Medium  
**Time Complexity**: O(n)  
**Space Complexity**: O(1)

```python
# Maximum Subarray Sum (Kadane's Algorithm)
class Solution:
    def maxSubArray(self, nums):
        """Find the contiguous subarray with maximum sum."""
        if not nums:
            return 0
        
        # Kadane's Algorithm
        max_sum = current_sum = nums[0]
        
        for i in range(1, len(nums)):
            current_sum = max(nums[i], current_sum + nums[i])
            max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    def maxSubArrayWithIndices(self, nums):
        """Find maximum subarray sum with start and end indices."""
        if not nums:
            return 0, 0, 0
        
        max_sum = current_sum = nums[0]
        start = end = 0
        temp_start = 0
        
        for i in range(1, len(nums)):
            if current_sum < 0:
                current_sum = nums[i]
                temp_start = i
            else:
                current_sum += nums[i]
            
            if current_sum > max_sum:
                max_sum = current_sum
                start = temp_start
                end = i
        
        return max_sum, start, end
    
    def maxSubArrayCircular(self, nums):
        """Find maximum subarray sum in circular array."""
        if not nums:
            return 0
        
        # Case 1: Maximum subarray is not circular
        max_non_circular = self.maxSubArray(nums)
        
        # Case 2: Maximum subarray is circular
        total_sum = sum(nums)
        min_subarray = self.minSubArray(nums)
        max_circular = total_sum - min_subarray
        
        # Handle edge case where all elements are negative
        if max_circular == 0:
            return max_non_circular
        
        return max(max_non_circular, max_circular)
    
    def minSubArray(self, nums):
        """Find minimum subarray sum."""
        min_sum = current_sum = nums[0]
        
        for i in range(1, len(nums)):
            current_sum = min(nums[i], current_sum + nums[i])
            min_sum = min(min_sum, current_sum)
        
        return min_sum
```

## 🔧 Graph Problems

### Problem 6: Course Schedule
**Difficulty**: Medium  
**Time Complexity**: O(V + E)  
**Space Complexity**: O(V + E)

```python
# Course Schedule (Topological Sort)
class Solution:
    def canFinish(self, numCourses, prerequisites):
        """Check if all courses can be finished."""
        # Build adjacency list
        graph = [[] for _ in range(numCourses)]
        in_degree = [0] * numCourses
        
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        
        # BFS with queue
        queue = []
        for i in range(numCourses):
            if in_degree[i] == 0:
                queue.append(i)
        
        completed = 0
        while queue:
            course = queue.pop(0)
            completed += 1
            
            for next_course in graph[course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)
        
        return completed == numCourses
    
    def findOrder(self, numCourses, prerequisites):
        """Find the order of courses to take."""
        graph = [[] for _ in range(numCourses)]
        in_degree = [0] * numCourses
        
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        
        queue = []
        for i in range(numCourses):
            if in_degree[i] == 0:
                queue.append(i)
        
        result = []
        while queue:
            course = queue.pop(0)
            result.append(course)
            
            for next_course in graph[course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)
        
        return result if len(result) == numCourses else []
    
    def canFinishDFS(self, numCourses, prerequisites):
        """DFS approach to detect cycle."""
        graph = [[] for _ in range(numCourses)]
        for course, prereq in prerequisites:
            graph[prereq].append(course)
        
        # 0: unvisited, 1: visiting, 2: visited
        state = [0] * numCourses
        
        def hasCycle(course):
            if state[course] == 1:  # Currently visiting
                return True
            if state[course] == 2:  # Already visited
                return False
            
            state[course] = 1  # Mark as visiting
            for next_course in graph[course]:
                if hasCycle(next_course):
                    return True
            state[course] = 2  # Mark as visited
            return False
        
        for course in range(numCourses):
            if state[course] == 0 and hasCycle(course):
                return False
        
        return True
```

### Problem 7: Word Ladder
**Difficulty**: Hard  
**Time Complexity**: O(M^2 * N)  
**Space Complexity**: O(M^2 * N)

```python
# Word Ladder
class Solution:
    def ladderLength(self, beginWord, endWord, wordList):
        """Find the shortest transformation sequence length."""
        if endWord not in wordList:
            return 0
        
        wordList = set(wordList)
        wordList.add(beginWord)
        
        # Build adjacency list
        graph = self.buildGraph(wordList)
        
        # BFS to find shortest path
        queue = [(beginWord, 1)]
        visited = {beginWord}
        
        while queue:
            word, length = queue.pop(0)
            
            if word == endWord:
                return length
            
            # Generate all possible one-character changes
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c != word[i]:
                        new_word = word[:i] + c + word[i+1:]
                        if new_word in wordList and new_word not in visited:
                            visited.add(new_word)
                            queue.append((new_word, length + 1))
        
        return 0
    
    def buildGraph(self, wordList):
        """Build graph where words are connected if they differ by one character."""
        graph = {}
        for word in wordList:
            graph[word] = []
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c != word[i]:
                        new_word = word[:i] + c + word[i+1:]
                        if new_word in wordList:
                            graph[word].append(new_word)
        return graph
    
    def findLadders(self, beginWord, endWord, wordList):
        """Find all shortest transformation sequences."""
        if endWord not in wordList:
            return []
        
        wordList = set(wordList)
        wordList.add(beginWord)
        
        # BFS to find shortest distance
        distances = {}
        distances[beginWord] = 0
        queue = [beginWord]
        
        while queue:
            word = queue.pop(0)
            if word == endWord:
                break
            
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c != word[i]:
                        new_word = word[:i] + c + word[i+1:]
                        if new_word in wordList and new_word not in distances:
                            distances[new_word] = distances[word] + 1
                            queue.append(new_word)
        
        # DFS to find all paths
        result = []
        self.dfs(beginWord, endWord, wordList, distances, [beginWord], result)
        return result
    
    def dfs(self, current, target, wordList, distances, path, result):
        """DFS to find all shortest paths."""
        if current == target:
            result.append(path[:])
            return
        
        for i in range(len(current)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != current[i]:
                    new_word = current[:i] + c + current[i+1:]
                    if (new_word in wordList and 
                        new_word in distances and 
                        distances[new_word] == distances[current] + 1):
                        path.append(new_word)
                        self.dfs(new_word, target, wordList, distances, path, result)
                        path.pop()
```

## 🎯 Best Practices

### Problem-Solving Approach
1. **Understand the Problem**: Read carefully and ask clarifying questions
2. **Identify Patterns**: Look for common algorithmic patterns
3. **Start Simple**: Begin with brute force, then optimize
4. **Consider Edge Cases**: Think about boundary conditions
5. **Test Your Solution**: Verify with examples

### Common Patterns
1. **Two Pointers**: For array and string problems
2. **Sliding Window**: For substring and subarray problems
3. **Hash Map**: For frequency counting and lookups
4. **Stack/Queue**: For parsing and traversal problems
5. **Dynamic Programming**: For optimization problems
6. **Graph Algorithms**: For connectivity and path problems

### Optimization Techniques
1. **Time Complexity**: Analyze and optimize time complexity
2. **Space Complexity**: Consider space-time tradeoffs
3. **Data Structures**: Choose appropriate data structures
4. **Algorithms**: Select efficient algorithms
5. **Edge Cases**: Handle all edge cases efficiently

---

**Last Updated**: December 2024  
**Category**: Advanced LeetCode Problems  
**Complexity**: Expert Level
