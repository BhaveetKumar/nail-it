# Find Peak Element

### Problem
A peak element is an element that is strictly greater than its neighbors.

Given a 0-indexed integer array `nums`, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.

You may imagine that `nums[-1] = nums[n] = -∞`.

You must write an algorithm that runs in `O(log n)` time.

**Example:**
```
Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.

Input: nums = [1,2,1,3,5,6,4]
Output: 5
Explanation: Your function can return either index number 1 where the peak element is 2, or index number 5 where the peak element is 6.
```

### Golang Solution

```go
func findPeakElement(nums []int) int {
    left, right := 0, len(nums)-1
    
    for left < right {
        mid := left + (right-left)/2
        
        if nums[mid] > nums[mid+1] {
            right = mid
        } else {
            left = mid + 1
        }
    }
    
    return left
}
```

### Alternative Solutions

#### **Linear Search**
```go
func findPeakElementLinear(nums []int) int {
    for i := 0; i < len(nums)-1; i++ {
        if nums[i] > nums[i+1] {
            return i
        }
    }
    return len(nums) - 1
}
```

#### **Return All Peaks**
```go
func findAllPeaks(nums []int) []int {
    var peaks []int
    
    for i := 0; i < len(nums); i++ {
        isPeak := true
        
        if i > 0 && nums[i] <= nums[i-1] {
            isPeak = false
        }
        
        if i < len(nums)-1 && nums[i] <= nums[i+1] {
            isPeak = false
        }
        
        if isPeak {
            peaks = append(peaks, i)
        }
    }
    
    return peaks
}
```

#### **Return with Peak Info**
```go
type PeakInfo struct {
    Index    int
    Value    int
    IsGlobal bool
    LeftNeighbor int
    RightNeighbor int
}

func findPeakWithInfo(nums []int) PeakInfo {
    if len(nums) == 0 {
        return PeakInfo{Index: -1}
    }
    
    if len(nums) == 1 {
        return PeakInfo{
            Index:    0,
            Value:    nums[0],
            IsGlobal: true,
        }
    }
    
    left, right := 0, len(nums)-1
    
    for left < right {
        mid := left + (right-left)/2
        
        if nums[mid] > nums[mid+1] {
            right = mid
        } else {
            left = mid + 1
        }
    }
    
    peakIndex := left
    peakValue := nums[peakIndex]
    
    // Check if it's a global peak
    isGlobal := true
    for _, num := range nums {
        if num > peakValue {
            isGlobal = false
            break
        }
    }
    
    leftNeighbor := -1
    if peakIndex > 0 {
        leftNeighbor = nums[peakIndex-1]
    }
    
    rightNeighbor := -1
    if peakIndex < len(nums)-1 {
        rightNeighbor = nums[peakIndex+1]
    }
    
    return PeakInfo{
        Index:         peakIndex,
        Value:         peakValue,
        IsGlobal:      isGlobal,
        LeftNeighbor:  leftNeighbor,
        RightNeighbor: rightNeighbor,
    }
}
```

#### **Return Peak Statistics**
```go
type PeakStats struct {
    TotalPeaks    int
    GlobalPeaks   int
    LocalPeaks    int
    MaxPeak       int
    MinPeak       int
    AvgPeak       float64
    PeakIndices   []int
    PeakValues    []int
}

func peakStatistics(nums []int) PeakStats {
    if len(nums) == 0 {
        return PeakStats{}
    }
    
    var peakIndices []int
    var peakValues []int
    
    for i := 0; i < len(nums); i++ {
        isPeak := true
        
        if i > 0 && nums[i] <= nums[i-1] {
            isPeak = false
        }
        
        if i < len(nums)-1 && nums[i] <= nums[i+1] {
            isPeak = false
        }
        
        if isPeak {
            peakIndices = append(peakIndices, i)
            peakValues = append(peakValues, nums[i])
        }
    }
    
    if len(peakValues) == 0 {
        return PeakStats{PeakIndices: peakIndices, PeakValues: peakValues}
    }
    
    maxPeak := peakValues[0]
    minPeak := peakValues[0]
    sum := 0
    
    for _, value := range peakValues {
        if value > maxPeak {
            maxPeak = value
        }
        if value < minPeak {
            minPeak = value
        }
        sum += value
    }
    
    // Count global peaks
    globalPeaks := 0
    maxValue := nums[0]
    for _, num := range nums {
        if num > maxValue {
            maxValue = num
        }
    }
    
    for _, value := range peakValues {
        if value == maxValue {
            globalPeaks++
        }
    }
    
    return PeakStats{
        TotalPeaks:  len(peakValues),
        GlobalPeaks: globalPeaks,
        LocalPeaks:  len(peakValues) - globalPeaks,
        MaxPeak:     maxPeak,
        MinPeak:     minPeak,
        AvgPeak:     float64(sum) / float64(len(peakValues)),
        PeakIndices: peakIndices,
        PeakValues:  peakValues,
    }
}
```

#### **Return Peak with Neighbors**
```go
type PeakWithNeighbors struct {
    Index    int
    Value    int
    Left     int
    Right    int
    Slope    string
    IsValid  bool
}

func findPeakWithNeighbors(nums []int) PeakWithNeighbors {
    if len(nums) == 0 {
        return PeakWithNeighbors{IsValid: false}
    }
    
    if len(nums) == 1 {
        return PeakWithNeighbors{
            Index:   0,
            Value:   nums[0],
            Left:    -1,
            Right:   -1,
            Slope:   "single",
            IsValid: true,
        }
    }
    
    left, right := 0, len(nums)-1
    
    for left < right {
        mid := left + (right-left)/2
        
        if nums[mid] > nums[mid+1] {
            right = mid
        } else {
            left = mid + 1
        }
    }
    
    peakIndex := left
    peakValue := nums[peakIndex]
    
    leftNeighbor := -1
    if peakIndex > 0 {
        leftNeighbor = nums[peakIndex-1]
    }
    
    rightNeighbor := -1
    if peakIndex < len(nums)-1 {
        rightNeighbor = nums[peakIndex+1]
    }
    
    // Determine slope
    slope := "peak"
    if leftNeighbor != -1 && rightNeighbor != -1 {
        if leftNeighbor < peakValue && peakValue > rightNeighbor {
            slope = "peak"
        } else if leftNeighbor < peakValue && peakValue < rightNeighbor {
            slope = "ascending"
        } else if leftNeighbor > peakValue && peakValue > rightNeighbor {
            slope = "descending"
        }
    } else if leftNeighbor == -1 {
        slope = "left_boundary"
    } else if rightNeighbor == -1 {
        slope = "right_boundary"
    }
    
    return PeakWithNeighbors{
        Index:   peakIndex,
        Value:   peakValue,
        Left:    leftNeighbor,
        Right:   rightNeighbor,
        Slope:   slope,
        IsValid: true,
    }
}
```

### Complexity
- **Time Complexity:** O(log n) for binary search, O(n) for linear search
- **Space Complexity:** O(1) for binary search, O(n) for storing all peaks
- **Space Complexity:** O(1) for iterative, O(log n) for recursive