# Assign Cookies

### Problem
Assume you are an awesome parent and want to give your children some cookies. But, you should give each child at most one cookie.

Each child `i` has a greed factor `g[i]`, which is the minimum size of a cookie that the child will be content with; and each cookie `j` has a size `s[j]`. If `s[j] >= g[i]`, we can assign the cookie `j` to the child `i`, and the child `i` will be content. Your goal is to maximize the number of your content children and output the maximum number.

**Example:**
```
Input: g = [1,2,3], s = [1,1]
Output: 1
Explanation: You have 3 children and 2 cookies. The greed factors of 3 children are 1, 2, 3. 
And even though you have 2 cookies, since their size is both 1, you could only make the child whose greed factor is 1 content.
You need to output 1.

Input: g = [1,2], s = [1,2,3]
Output: 2
Explanation: You have 2 children and 3 cookies. The greed factors of 2 children are 1, 2. 
You have 3 cookies and their sizes are big enough to gratify all of the children, 
You need to output 2.
```

### Golang Solution

```go
import "sort"

func findContentChildren(g []int, s []int) int {
    sort.Ints(g)
    sort.Ints(s)
    
    childIndex := 0
    cookieIndex := 0
    contentChildren := 0
    
    for childIndex < len(g) && cookieIndex < len(s) {
        if s[cookieIndex] >= g[childIndex] {
            contentChildren++
            childIndex++
        }
        cookieIndex++
    }
    
    return contentChildren
}
```

### Alternative Solutions

#### **Greedy with Priority**
```go
import "sort"

func findContentChildrenPriority(g []int, s []int) int {
    sort.Ints(g)
    sort.Ints(s)
    
    contentChildren := 0
    cookieIndex := 0
    
    for _, greed := range g {
        for cookieIndex < len(s) && s[cookieIndex] < greed {
            cookieIndex++
        }
        
        if cookieIndex < len(s) {
            contentChildren++
            cookieIndex++
        }
    }
    
    return contentChildren
}
```

#### **Return Assignment Details**
```go
type AssignmentResult struct {
    ContentChildren int
    Assignments     []Assignment
    UnassignedChildren []int
    UnusedCookies   []int
}

type Assignment struct {
    ChildIndex int
    CookieIndex int
    Greed      int
    CookieSize int
}

func findContentChildrenWithDetails(g []int, s []int) AssignmentResult {
    sort.Ints(g)
    sort.Ints(s)
    
    var assignments []Assignment
    var unassignedChildren []int
    var unusedCookies []int
    
    childIndex := 0
    cookieIndex := 0
    
    for childIndex < len(g) && cookieIndex < len(s) {
        if s[cookieIndex] >= g[childIndex] {
            assignments = append(assignments, Assignment{
                ChildIndex:  childIndex,
                CookieIndex: cookieIndex,
                Greed:       g[childIndex],
                CookieSize:  s[cookieIndex],
            })
            childIndex++
        }
        cookieIndex++
    }
    
    // Add unassigned children
    for i := childIndex; i < len(g); i++ {
        unassignedChildren = append(unassignedChildren, g[i])
    }
    
    // Add unused cookies
    for i := cookieIndex; i < len(s); i++ {
        unusedCookies = append(unusedCookies, s[i])
    }
    
    return AssignmentResult{
        ContentChildren:    len(assignments),
        Assignments:       assignments,
        UnassignedChildren: unassignedChildren,
        UnusedCookies:     unusedCookies,
    }
}
```

#### **Return All Possible Assignments**
```go
func findAllAssignments(g []int, s []int) [][]Assignment {
    var allAssignments [][]Assignment
    
    var backtrack func(int, []Assignment, []bool)
    backtrack = func(childIndex int, current []Assignment, used []bool) {
        if childIndex == len(g) {
            assignment := make([]Assignment, len(current))
            copy(assignment, current)
            allAssignments = append(allAssignments, assignment)
            return
        }
        
        for cookieIndex := 0; cookieIndex < len(s); cookieIndex++ {
            if !used[cookieIndex] && s[cookieIndex] >= g[childIndex] {
                used[cookieIndex] = true
                current = append(current, Assignment{
                    ChildIndex:  childIndex,
                    CookieIndex: cookieIndex,
                    Greed:       g[childIndex],
                    CookieSize:  s[cookieIndex],
                })
                
                backtrack(childIndex+1, current, used)
                
                current = current[:len(current)-1]
                used[cookieIndex] = false
            }
        }
    }
    
    used := make([]bool, len(s))
    backtrack(0, []Assignment{}, used)
    
    return allAssignments
}
```

#### **Return Assignment Statistics**
```go
type AssignmentStats struct {
    TotalChildren      int
    TotalCookies       int
    ContentChildren    int
    UnassignedChildren int
    UnusedCookies      int
    MinGreed           int
    MaxGreed           int
    MinCookieSize      int
    MaxCookieSize      int
    AvgGreed           float64
    AvgCookieSize      float64
}

func assignmentStatistics(g []int, s []int) AssignmentStats {
    if len(g) == 0 || len(s) == 0 {
        return AssignmentStats{}
    }
    
    result := findContentChildrenWithDetails(g, s)
    
    minGreed := g[0]
    maxGreed := g[0]
    sumGreed := 0
    
    for _, greed := range g {
        if greed < minGreed {
            minGreed = greed
        }
        if greed > maxGreed {
            maxGreed = greed
        }
        sumGreed += greed
    }
    
    minCookieSize := s[0]
    maxCookieSize := s[0]
    sumCookieSize := 0
    
    for _, size := range s {
        if size < minCookieSize {
            minCookieSize = size
        }
        if size > maxCookieSize {
            maxCookieSize = size
        }
        sumCookieSize += size
    }
    
    return AssignmentStats{
        TotalChildren:      len(g),
        TotalCookies:       len(s),
        ContentChildren:    result.ContentChildren,
        UnassignedChildren: len(result.UnassignedChildren),
        UnusedCookies:      len(result.UnusedCookies),
        MinGreed:           minGreed,
        MaxGreed:           maxGreed,
        MinCookieSize:      minCookieSize,
        MaxCookieSize:      maxCookieSize,
        AvgGreed:           float64(sumGreed) / float64(len(g)),
        AvgCookieSize:      float64(sumCookieSize) / float64(len(s)),
    }
}
```

#### **Return Optimal Assignment**
```go
func findOptimalAssignment(g []int, s []int) AssignmentResult {
    sort.Ints(g)
    sort.Ints(s)
    
    var assignments []Assignment
    var unassignedChildren []int
    var unusedCookies []int
    
    childIndex := 0
    cookieIndex := 0
    
    for childIndex < len(g) && cookieIndex < len(s) {
        if s[cookieIndex] >= g[childIndex] {
            assignments = append(assignments, Assignment{
                ChildIndex:  childIndex,
                CookieIndex: cookieIndex,
                Greed:       g[childIndex],
                CookieSize:  s[cookieIndex],
            })
            childIndex++
        }
        cookieIndex++
    }
    
    // Add unassigned children
    for i := childIndex; i < len(g); i++ {
        unassignedChildren = append(unassignedChildren, g[i])
    }
    
    // Add unused cookies
    for i := cookieIndex; i < len(s); i++ {
        unusedCookies = append(unusedCookies, s[i])
    }
    
    return AssignmentResult{
        ContentChildren:    len(assignments),
        Assignments:       assignments,
        UnassignedChildren: unassignedChildren,
        UnusedCookies:     unusedCookies,
    }
}
```

### Complexity
- **Time Complexity:** O(n log n + m log m) where n and m are lengths of arrays
- **Space Complexity:** O(1) for in-place, O(n + m) for additional arrays