---
# Auto-generated front matter
Title: Courseschedule
LastUpdated: 2025-11-06T20:45:58.743860
Tags: []
Status: draft
---

# Course Schedule

### Problem
There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [ai, bi]` indicates that you must take course `bi` first if you want to take course `ai`.

For example, the pair `[0, 1]`, indicates that to take course `0` you have to first take course `1`.

Return `true` if you can finish all courses. Otherwise, return `false`.

**Example:**
```
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0. So it is possible.

Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.
```

### Golang Solution

```go
func canFinish(numCourses int, prerequisites [][]int) bool {
    // Build adjacency list
    graph := make([][]int, numCourses)
    inDegree := make([]int, numCourses)
    
    for _, prereq := range prerequisites {
        course, prereq := prereq[0], prereq[1]
        graph[prereq] = append(graph[prereq], course)
        inDegree[course]++
    }
    
    // Find courses with no prerequisites
    queue := []int{}
    for i := 0; i < numCourses; i++ {
        if inDegree[i] == 0 {
            queue = append(queue, i)
        }
    }
    
    completed := 0
    
    // Process courses
    for len(queue) > 0 {
        course := queue[0]
        queue = queue[1:]
        completed++
        
        // Reduce in-degree for dependent courses
        for _, dependent := range graph[course] {
            inDegree[dependent]--
            if inDegree[dependent] == 0 {
                queue = append(queue, dependent)
            }
        }
    }
    
    return completed == numCourses
}
```

### Alternative Solutions

#### **Using DFS (Detect Cycle)**
```go
func canFinishDFS(numCourses int, prerequisites [][]int) bool {
    // Build adjacency list
    graph := make([][]int, numCourses)
    
    for _, prereq := range prerequisites {
        course, prereq := prereq[0], prereq[1]
        graph[prereq] = append(graph[prereq], course)
    }
    
    // 0: unvisited, 1: visiting, 2: visited
    state := make([]int, numCourses)
    
    var dfs func(int) bool
    dfs = func(course int) bool {
        if state[course] == 1 {
            return false // Cycle detected
        }
        if state[course] == 2 {
            return true // Already processed
        }
        
        state[course] = 1 // Mark as visiting
        
        for _, dependent := range graph[course] {
            if !dfs(dependent) {
                return false
            }
        }
        
        state[course] = 2 // Mark as visited
        return true
    }
    
    for i := 0; i < numCourses; i++ {
        if state[i] == 0 {
            if !dfs(i) {
                return false
            }
        }
    }
    
    return true
}
```

#### **Return Course Order**
```go
func findOrder(numCourses int, prerequisites [][]int) []int {
    // Build adjacency list
    graph := make([][]int, numCourses)
    inDegree := make([]int, numCourses)
    
    for _, prereq := range prerequisites {
        course, prereq := prereq[0], prereq[1]
        graph[prereq] = append(graph[prereq], course)
        inDegree[course]++
    }
    
    // Find courses with no prerequisites
    queue := []int{}
    for i := 0; i < numCourses; i++ {
        if inDegree[i] == 0 {
            queue = append(queue, i)
        }
    }
    
    var result []int
    
    // Process courses
    for len(queue) > 0 {
        course := queue[0]
        queue = queue[1:]
        result = append(result, course)
        
        // Reduce in-degree for dependent courses
        for _, dependent := range graph[course] {
            inDegree[dependent]--
            if inDegree[dependent] == 0 {
                queue = append(queue, dependent)
            }
        }
    }
    
    if len(result) != numCourses {
        return []int{} // Cycle detected
    }
    
    return result
}
```

#### **Return All Possible Orders**
```go
func findAllOrders(numCourses int, prerequisites [][]int) [][]int {
    // Build adjacency list
    graph := make([][]int, numCourses)
    inDegree := make([]int, numCourses)
    
    for _, prereq := range prerequisites {
        course, prereq := prereq[0], prereq[1]
        graph[prereq] = append(graph[prereq], course)
        inDegree[course]++
    }
    
    var result [][]int
    var current []int
    visited := make([]bool, numCourses)
    
    var backtrack func()
    backtrack = func() {
        if len(current) == numCourses {
            order := make([]int, len(current))
            copy(order, current)
            result = append(result, order)
            return
        }
        
        for i := 0; i < numCourses; i++ {
            if !visited[i] && inDegree[i] == 0 {
                visited[i] = true
                current = append(current, i)
                
                // Update in-degrees
                for _, dependent := range graph[i] {
                    inDegree[dependent]--
                }
                
                backtrack()
                
                // Restore in-degrees
                for _, dependent := range graph[i] {
                    inDegree[dependent]++
                }
                
                current = current[:len(current)-1]
                visited[i] = false
            }
        }
    }
    
    backtrack()
    return result
}
```

#### **Return Course Statistics**
```go
type CourseStats struct {
    CanFinish    bool
    TotalCourses int
    Completed    int
    Remaining    int
    Prerequisites map[int][]int
}

func courseStatistics(numCourses int, prerequisites [][]int) CourseStats {
    // Build adjacency list
    graph := make([][]int, numCourses)
    inDegree := make([]int, numCourses)
    prereqMap := make(map[int][]int)
    
    for _, prereq := range prerequisites {
        course, prereq := prereq[0], prereq[1]
        graph[prereq] = append(graph[prereq], course)
        inDegree[course]++
        prereqMap[course] = append(prereqMap[course], prereq)
    }
    
    // Find courses with no prerequisites
    queue := []int{}
    for i := 0; i < numCourses; i++ {
        if inDegree[i] == 0 {
            queue = append(queue, i)
        }
    }
    
    completed := 0
    
    // Process courses
    for len(queue) > 0 {
        course := queue[0]
        queue = queue[1:]
        completed++
        
        // Reduce in-degree for dependent courses
        for _, dependent := range graph[course] {
            inDegree[dependent]--
            if inDegree[dependent] == 0 {
                queue = append(queue, dependent)
            }
        }
    }
    
    return CourseStats{
        CanFinish:     completed == numCourses,
        TotalCourses:  numCourses,
        Completed:     completed,
        Remaining:     numCourses - completed,
        Prerequisites: prereqMap,
    }
}
```

### Complexity
- **Time Complexity:** O(V + E) where V is number of courses and E is number of prerequisites
- **Space Complexity:** O(V + E)