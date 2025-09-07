# Course Schedule

### Problem
There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [ai, bi]` indicates that you must take course `bi` first if you want to take course `ai`.

Return `true` if you can finish all courses. Otherwise, return `false`.

**Example:**
```
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished course 0. So it is possible.

Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.
```

### Golang Solution

```go
func canFinish(numCourses int, prerequisites [][]int) bool {
    graph := make(map[int][]int)
    inDegree := make([]int, numCourses)
    
    // Build graph and calculate in-degrees
    for _, prereq := range prerequisites {
        course, prereqCourse := prereq[0], prereq[1]
        graph[prereqCourse] = append(graph[prereqCourse], course)
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

### Complexity
- **Time Complexity:** O(V + E) where V is courses, E is prerequisites
- **Space Complexity:** O(V + E)
