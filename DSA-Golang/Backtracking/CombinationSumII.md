# Combination Sum II

### Problem
Given a collection of candidate numbers (`candidates`) and a target number (`target`), find all unique combinations in `candidates` where the candidate numbers sum to `target`.

Each number in `candidates` may only be used once in the combination.

Note: The solution set must not contain duplicate combinations.

**Example:**
```
Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]

Input: candidates = [2,5,2,1,2], target = 5
Output: 
[
[1,2,2],
[5]
]
```

### Golang Solution

```go
import "sort"

func combinationSum2(candidates []int, target int) [][]int {
    sort.Ints(candidates)
    var result [][]int
    var current []int
    
    var backtrack func(int, int)
    backtrack = func(start int, remaining int) {
        if remaining == 0 {
            combination := make([]int, len(current))
            copy(combination, current)
            result = append(result, combination)
            return
        }
        
        for i := start; i < len(candidates); i++ {
            if i > start && candidates[i] == candidates[i-1] {
                continue
            }
            
            if candidates[i] > remaining {
                break
            }
            
            current = append(current, candidates[i])
            backtrack(i+1, remaining-candidates[i])
            current = current[:len(current)-1]
        }
    }
    
    backtrack(0, target)
    return result
}
```

### Alternative Solutions

#### **Using Set to Avoid Duplicates**
```go
import "sort"

func combinationSum2Set(candidates []int, target int) [][]int {
    sort.Ints(candidates)
    var result [][]int
    var current []int
    seen := make(map[string]bool)
    
    var backtrack func(int, int)
    backtrack = func(start int, remaining int) {
        if remaining == 0 {
            key := fmt.Sprintf("%v", current)
            if !seen[key] {
                combination := make([]int, len(current))
                copy(combination, current)
                result = append(result, combination)
                seen[key] = true
            }
            return
        }
        
        for i := start; i < len(candidates); i++ {
            if candidates[i] > remaining {
                break
            }
            
            current = append(current, candidates[i])
            backtrack(i+1, remaining-candidates[i])
            current = current[:len(current)-1]
        }
    }
    
    backtrack(0, target)
    return result
}
```

#### **Return with Counts**
```go
type CombinationResult struct {
    Combinations [][]int
    Count        int
    MinLength    int
    MaxLength    int
}

func combinationSum2WithCounts(candidates []int, target int) CombinationResult {
    sort.Ints(candidates)
    var combinations [][]int
    var current []int
    
    var backtrack func(int, int)
    backtrack = func(start int, remaining int) {
        if remaining == 0 {
            combination := make([]int, len(current))
            copy(combination, current)
            combinations = append(combinations, combination)
            return
        }
        
        for i := start; i < len(candidates); i++ {
            if i > start && candidates[i] == candidates[i-1] {
                continue
            }
            
            if candidates[i] > remaining {
                break
            }
            
            current = append(current, candidates[i])
            backtrack(i+1, remaining-candidates[i])
            current = current[:len(current)-1]
        }
    }
    
    backtrack(0, target)
    
    if len(combinations) == 0 {
        return CombinationResult{Combinations: combinations, Count: 0}
    }
    
    minLength := len(combinations[0])
    maxLength := len(combinations[0])
    
    for _, combo := range combinations {
        if len(combo) < minLength {
            minLength = len(combo)
        }
        if len(combo) > maxLength {
            maxLength = len(combo)
        }
    }
    
    return CombinationResult{
        Combinations: combinations,
        Count:        len(combinations),
        MinLength:    minLength,
        MaxLength:    maxLength,
    }
}
```

#### **Return All Possible Sums**
```go
func allPossibleSums(candidates []int) []int {
    var sums []int
    seen := make(map[int]bool)
    
    var backtrack func(int, int)
    backtrack = func(start int, currentSum int) {
        if currentSum > 0 && !seen[currentSum] {
            sums = append(sums, currentSum)
            seen[currentSum] = true
        }
        
        for i := start; i < len(candidates); i++ {
            backtrack(i+1, currentSum+candidates[i])
        }
    }
    
    backtrack(0, 0)
    return sums
}
```

#### **Return with Statistics**
```go
type CombinationStats struct {
    TotalCombinations int
    MinLength        int
    MaxLength        int
    AvgLength        float64
    MinSum           int
    MaxSum           int
    AvgSum           float64
    MostFrequentElement int
    ElementFrequency map[int]int
}

func combinationSum2Stats(candidates []int, target int) CombinationStats {
    combinations := combinationSum2(candidates, target)
    
    if len(combinations) == 0 {
        return CombinationStats{ElementFrequency: make(map[int]int)}
    }
    
    minLength := len(combinations[0])
    maxLength := len(combinations[0])
    totalLength := 0
    totalSum := 0
    elementFreq := make(map[int]int)
    
    for _, combo := range combinations {
        length := len(combo)
        totalLength += length
        
        if length < minLength {
            minLength = length
        }
        if length > maxLength {
            maxLength = length
        }
        
        sum := 0
        for _, num := range combo {
            sum += num
            elementFreq[num]++
        }
        totalSum += sum
    }
    
    mostFrequent := 0
    maxFreq := 0
    for element, freq := range elementFreq {
        if freq > maxFreq {
            maxFreq = freq
            mostFrequent = element
        }
    }
    
    return CombinationStats{
        TotalCombinations:  len(combinations),
        MinLength:         minLength,
        MaxLength:         maxLength,
        AvgLength:         float64(totalLength) / float64(len(combinations)),
        MinSum:            target,
        MaxSum:            target,
        AvgSum:            float64(totalSum) / float64(len(combinations)),
        MostFrequentElement: mostFrequent,
        ElementFrequency:   elementFreq,
    }
}
```

#### **Return with Path Info**
```go
type CombinationPath struct {
    Combination []int
    Path        []int
    Sum         int
    Length      int
}

func combinationSum2WithPaths(candidates []int, target int) []CombinationPath {
    sort.Ints(candidates)
    var result []CombinationPath
    var current []int
    var path []int
    
    var backtrack func(int, int)
    backtrack = func(start int, remaining int) {
        if remaining == 0 {
            sum := 0
            for _, num := range current {
                sum += num
            }
            
            result = append(result, CombinationPath{
                Combination: append([]int{}, current...),
                Path:        append([]int{}, path...),
                Sum:         sum,
                Length:      len(current),
            })
            return
        }
        
        for i := start; i < len(candidates); i++ {
            if i > start && candidates[i] == candidates[i-1] {
                continue
            }
            
            if candidates[i] > remaining {
                break
            }
            
            current = append(current, candidates[i])
            path = append(path, i)
            backtrack(i+1, remaining-candidates[i])
            current = current[:len(current)-1]
            path = path[:len(path)-1]
        }
    }
    
    backtrack(0, target)
    return result
}
```

### Complexity
- **Time Complexity:** O(2^n) in worst case, O(n log n) for sorting
- **Space Complexity:** O(target) for recursion stack