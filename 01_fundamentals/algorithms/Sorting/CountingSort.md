---
# Auto-generated front matter
Title: Countingsort
LastUpdated: 2025-11-06T20:45:58.734941
Tags: []
Status: draft
---

# Counting Sort

### Problem
Implement Counting Sort algorithm to sort an array of integers.

Counting Sort is a non-comparison based sorting algorithm that works by counting the number of objects having distinct key values.

**Example:**
```
Input: [4, 2, 2, 8, 3, 3, 1]
Output: [1, 2, 2, 3, 3, 4, 8]
```

### Golang Solution

```go
func countingSort(nums []int) []int {
    if len(nums) == 0 {
        return nums
    }
    
    // Find the range of input array
    minVal := nums[0]
    maxVal := nums[0]
    
    for _, num := range nums {
        if num < minVal {
            minVal = num
        }
        if num > maxVal {
            maxVal = num
        }
    }
    
    // Create count array
    rangeSize := maxVal - minVal + 1
    count := make([]int, rangeSize)
    
    // Count occurrences of each element
    for _, num := range nums {
        count[num-minVal]++
    }
    
    // Modify count array to store actual position
    for i := 1; i < rangeSize; i++ {
        count[i] += count[i-1]
    }
    
    // Build output array
    output := make([]int, len(nums))
    for i := len(nums) - 1; i >= 0; i-- {
        output[count[nums[i]-minVal]-1] = nums[i]
        count[nums[i]-minVal]--
    }
    
    return output
}
```

### Alternative Solutions

#### **In-Place Counting Sort**
```go
func countingSortInPlace(nums []int) {
    if len(nums) == 0 {
        return
    }
    
    // Find the range of input array
    minVal := nums[0]
    maxVal := nums[0]
    
    for _, num := range nums {
        if num < minVal {
            minVal = num
        }
        if num > maxVal {
            maxVal = num
        }
    }
    
    // Create count array
    rangeSize := maxVal - minVal + 1
    count := make([]int, rangeSize)
    
    // Count occurrences of each element
    for _, num := range nums {
        count[num-minVal]++
    }
    
    // Reconstruct the array
    index := 0
    for i := 0; i < rangeSize; i++ {
        for j := 0; j < count[i]; j++ {
            nums[index] = i + minVal
            index++
        }
    }
}
```

#### **Counting Sort with Range**
```go
func countingSortWithRange(nums []int, minVal, maxVal int) []int {
    if len(nums) == 0 {
        return nums
    }
    
    rangeSize := maxVal - minVal + 1
    count := make([]int, rangeSize)
    
    // Count occurrences of each element
    for _, num := range nums {
        if num < minVal || num > maxVal {
            continue // Skip out of range elements
        }
        count[num-minVal]++
    }
    
    // Build output array
    output := make([]int, 0, len(nums))
    for i := 0; i < rangeSize; i++ {
        for j := 0; j < count[i]; j++ {
            output = append(output, i+minVal)
        }
    }
    
    return output
}
```

#### **Return with Statistics**
```go
type CountingSortStats struct {
    SortedArray    []int
    OriginalArray  []int
    MinValue       int
    MaxValue       int
    RangeSize      int
    Comparisons    int
    Swaps          int
    TimeComplexity string
    SpaceComplexity string
}

func countingSortWithStats(nums []int) CountingSortStats {
    originalArray := make([]int, len(nums))
    copy(originalArray, nums)
    
    if len(nums) == 0 {
        return CountingSortStats{
            SortedArray:     nums,
            OriginalArray:   originalArray,
            TimeComplexity:  "O(n+k)",
            SpaceComplexity: "O(k)",
        }
    }
    
    // Find the range of input array
    minVal := nums[0]
    maxVal := nums[0]
    
    for _, num := range nums {
        if num < minVal {
            minVal = num
        }
        if num > maxVal {
            maxVal = num
        }
    }
    
    rangeSize := maxVal - minVal + 1
    count := make([]int, rangeSize)
    
    // Count occurrences of each element
    for _, num := range nums {
        count[num-minVal]++
    }
    
    // Build output array
    output := make([]int, len(nums))
    for i := len(nums) - 1; i >= 0; i-- {
        output[count[nums[i]-minVal]-1] = nums[i]
        count[nums[i]-minVal]--
    }
    
    return CountingSortStats{
        SortedArray:     output,
        OriginalArray:   originalArray,
        MinValue:        minVal,
        MaxValue:        maxVal,
        RangeSize:       rangeSize,
        Comparisons:     0, // No comparisons in counting sort
        Swaps:           0, // No swaps in counting sort
        TimeComplexity:  "O(n+k)",
        SpaceComplexity: "O(k)",
    }
}
```

#### **Return with Frequency**
```go
type ElementFrequency struct {
    Value     int
    Frequency int
    Positions []int
}

type CountingSortWithFreq struct {
    SortedArray    []int
    Frequencies    []ElementFrequency
    UniqueElements int
    MostFrequent   int
    LeastFrequent  int
}

func countingSortWithFrequency(nums []int) CountingSortWithFreq {
    if len(nums) == 0 {
        return CountingSortWithFreq{
            SortedArray: nums,
            Frequencies: []ElementFrequency{},
        }
    }
    
    // Find the range of input array
    minVal := nums[0]
    maxVal := nums[0]
    
    for _, num := range nums {
        if num < minVal {
            minVal = num
        }
        if num > maxVal {
            maxVal = num
        }
    }
    
    rangeSize := maxVal - minVal + 1
    count := make([]int, rangeSize)
    positions := make(map[int][]int)
    
    // Count occurrences and track positions
    for i, num := range nums {
        count[num-minVal]++
        positions[num] = append(positions[num], i)
    }
    
    // Build output array
    output := make([]int, len(nums))
    for i := len(nums) - 1; i >= 0; i-- {
        output[count[nums[i]-minVal]-1] = nums[i]
        count[nums[i]-minVal]--
    }
    
    // Build frequency array
    var frequencies []ElementFrequency
    uniqueElements := 0
    mostFrequent := 0
    leastFrequent := len(nums)
    
    for i := 0; i < rangeSize; i++ {
        if count[i] > 0 {
            value := i + minVal
            freq := count[i]
            frequencies = append(frequencies, ElementFrequency{
                Value:     value,
                Frequency: freq,
                Positions: positions[value],
            })
            uniqueElements++
            
            if freq > mostFrequent {
                mostFrequent = freq
            }
            if freq < leastFrequent {
                leastFrequent = freq
            }
        }
    }
    
    return CountingSortWithFreq{
        SortedArray:    output,
        Frequencies:    frequencies,
        UniqueElements: uniqueElements,
        MostFrequent:   mostFrequent,
        LeastFrequent:  leastFrequent,
    }
}
```

#### **Return with Steps**
```go
type CountingSortStep struct {
    Step        string
    Array       []int
    CountArray  []int
    OutputArray []int
    Description string
}

func countingSortWithSteps(nums []int) []CountingSortStep {
    var steps []CountingSortStep
    originalArray := make([]int, len(nums))
    copy(originalArray, nums)
    
    if len(nums) == 0 {
        return steps
    }
    
    // Find the range of input array
    minVal := nums[0]
    maxVal := nums[0]
    
    for _, num := range nums {
        if num < minVal {
            minVal = num
        }
        if num > maxVal {
            maxVal = num
        }
    }
    
    rangeSize := maxVal - minVal + 1
    count := make([]int, rangeSize)
    
    steps = append(steps, CountingSortStep{
        Step:        "Initialize",
        Array:       append([]int{}, nums...),
        CountArray:  append([]int{}, count...),
        OutputArray: []int{},
        Description: fmt.Sprintf("Found range: %d to %d, size: %d", minVal, maxVal, rangeSize),
    })
    
    // Count occurrences of each element
    for _, num := range nums {
        count[num-minVal]++
    }
    
    steps = append(steps, CountingSortStep{
        Step:        "Count",
        Array:       append([]int{}, nums...),
        CountArray:  append([]int{}, count...),
        OutputArray: []int{},
        Description: "Counted occurrences of each element",
    })
    
    // Modify count array to store actual position
    for i := 1; i < rangeSize; i++ {
        count[i] += count[i-1]
    }
    
    steps = append(steps, CountingSortStep{
        Step:        "Cumulative",
        Array:       append([]int{}, nums...),
        CountArray:  append([]int{}, count...),
        OutputArray: []int{},
        Description: "Modified count array to store actual positions",
    })
    
    // Build output array
    output := make([]int, len(nums))
    for i := len(nums) - 1; i >= 0; i-- {
        output[count[nums[i]-minVal]-1] = nums[i]
        count[nums[i]-minVal]--
        
        steps = append(steps, CountingSortStep{
            Step:        fmt.Sprintf("Place %d", nums[i]),
            Array:       append([]int{}, nums...),
            CountArray:  append([]int{}, count...),
            OutputArray: append([]int{}, output...),
            Description: fmt.Sprintf("Placed %d at position %d", nums[i], count[nums[i]-minVal]),
        })
    }
    
    return steps
}
```

### Complexity
- **Time Complexity:** O(n+k) where n is the number of elements and k is the range of input
- **Space Complexity:** O(k) for the count array