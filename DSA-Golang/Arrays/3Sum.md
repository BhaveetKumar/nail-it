# 3Sum

### Problem
Given an integer array `nums`, return all the triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.

Notice that the solution set must not contain duplicate triplets.

**Example:**
```
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Explanation: 
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
The distinct triplets are [-1,0,1] and [-1,-1,2].
```

**Constraints:**
- 3 ≤ nums.length ≤ 3000
- -10⁵ ≤ nums[i] ≤ 10⁵

### Explanation

#### **Two Pointers Approach**
- Sort the array first
- Fix one element and use two pointers for the remaining two
- Skip duplicates to avoid duplicate triplets
- Time Complexity: O(n²)
- Space Complexity: O(1)

### Dry Run

**Input:** `nums = [-1,0,1,2,-1,-4]`

After sorting: `[-4,-1,-1,0,1,2]`

| i | left | right | sum | Action |
|---|------|-------|-----|---------|
| 0 | 1 | 5 | -4+(-1)+2 = -3 | sum < 0, left++ |
| 0 | 2 | 5 | -4+(-1)+2 = -3 | sum < 0, left++ |
| 0 | 3 | 5 | -4+0+2 = -2 | sum < 0, left++ |
| 0 | 4 | 5 | -4+1+2 = -1 | sum < 0, left++ |
| 1 | 2 | 5 | -1+(-1)+2 = 0 | **Found!** Add [-1,-1,2] |
| 1 | 3 | 4 | -1+0+1 = 0 | **Found!** Add [-1,0,1] |

**Result:** `[[-1,-1,2],[-1,0,1]]`

### Complexity
- **Time Complexity:** O(n²) - Two nested loops
- **Space Complexity:** O(1) - Only using constant extra space

### Golang Solution

```go
func threeSum(nums []int) [][]int {
    if len(nums) < 3 {
        return [][]int{}
    }
    
    // Sort the array
    sort.Ints(nums)
    var result [][]int
    
    for i := 0; i < len(nums)-2; i++ {
        // Skip duplicates for the first element
        if i > 0 && nums[i] == nums[i-1] {
            continue
        }
        
        left, right := i+1, len(nums)-1
        target := -nums[i]
        
        while left < right {
            sum := nums[left] + nums[right]
            
            if sum == target {
                result = append(result, []int{nums[i], nums[left], nums[right]})
                
                // Skip duplicates
                for left < right && nums[left] == nums[left+1] {
                    left++
                }
                for left < right && nums[right] == nums[right-1] {
                    right--
                }
                
                left++
                right--
            } else if sum < target {
                left++
            } else {
                right--
            }
        }
    }
    
    return result
}
```

### Notes / Variations

#### **Related Problems**
- **Two Sum**: Find two numbers that sum to target
- **4Sum**: Find four numbers that sum to target
- **3Sum Closest**: Find three numbers with sum closest to target
- **3Sum Smaller**: Count triplets with sum smaller than target

#### **ICPC Insights**
- **Sorting**: Always sort array for two pointers approach
- **Duplicate Handling**: Skip duplicates to avoid duplicate results
- **Early Termination**: Can optimize with early breaks
- **Two Pointers**: Classic technique for sorted array problems
