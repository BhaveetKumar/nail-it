---
# Auto-generated front matter
Title: Canplaceflowers
LastUpdated: 2025-11-06T20:45:58.731538
Tags: []
Status: draft
---

# Can Place Flowers

### Problem
You have a long flowerbed in which some of the plots are planted, and some are not. However, flowers cannot be planted in adjacent plots.

Given an integer array `flowerbed` containing 0's and 1's, where 0 means empty and 1 means not empty, and an integer `n`, return `true` if `n` new flowers can be planted in the flowerbed without violating the no-adjacent-flowers rule and `false` otherwise.

**Example:**
```
Input: flowerbed = [1,0,0,0,1], n = 1
Output: true

Input: flowerbed = [1,0,0,0,1], n = 2
Output: false
```

### Golang Solution

```go
func canPlaceFlowers(flowerbed []int, n int) bool {
    count := 0
    
    for i := 0; i < len(flowerbed) && count < n; i++ {
        if flowerbed[i] == 0 {
            prev := 0
            if i > 0 {
                prev = flowerbed[i-1]
            }
            
            next := 0
            if i < len(flowerbed)-1 {
                next = flowerbed[i+1]
            }
            
            if prev == 0 && next == 0 {
                flowerbed[i] = 1
                count++
            }
        }
    }
    
    return count >= n
}
```

### Alternative Solutions

#### **Using Greedy Approach**
```go
func canPlaceFlowersGreedy(flowerbed []int, n int) bool {
    if n == 0 {
        return true
    }
    
    for i := 0; i < len(flowerbed); i++ {
        if flowerbed[i] == 0 {
            canPlant := true
            
            // Check left neighbor
            if i > 0 && flowerbed[i-1] == 1 {
                canPlant = false
            }
            
            // Check right neighbor
            if i < len(flowerbed)-1 && flowerbed[i+1] == 1 {
                canPlant = false
            }
            
            if canPlant {
                flowerbed[i] = 1
                n--
                if n == 0 {
                    return true
                }
            }
        }
    }
    
    return false
}
```

#### **Return All Possible Positions**
```go
func findFlowerPositions(flowerbed []int, n int) (bool, []int) {
    var positions []int
    count := 0
    
    for i := 0; i < len(flowerbed) && count < n; i++ {
        if flowerbed[i] == 0 {
            prev := 0
            if i > 0 {
                prev = flowerbed[i-1]
            }
            
            next := 0
            if i < len(flowerbed)-1 {
                next = flowerbed[i+1]
            }
            
            if prev == 0 && next == 0 {
                flowerbed[i] = 1
                positions = append(positions, i)
                count++
            }
        }
    }
    
    return count >= n, positions
}
```

#### **Using Sliding Window**
```go
func canPlaceFlowersSlidingWindow(flowerbed []int, n int) bool {
    if n == 0 {
        return true
    }
    
    // Add padding to handle edge cases
    padded := make([]int, len(flowerbed)+2)
    copy(padded[1:], flowerbed)
    
    count := 0
    for i := 1; i < len(padded)-1; i++ {
        if padded[i] == 0 && padded[i-1] == 0 && padded[i+1] == 0 {
            padded[i] = 1
            count++
            if count >= n {
                return true
            }
        }
    }
    
    return false
}
```

#### **Return Maximum Flowers**
```go
func maxFlowersCanPlace(flowerbed []int) int {
    count := 0
    
    for i := 0; i < len(flowerbed); i++ {
        if flowerbed[i] == 0 {
            prev := 0
            if i > 0 {
                prev = flowerbed[i-1]
            }
            
            next := 0
            if i < len(flowerbed)-1 {
                next = flowerbed[i+1]
            }
            
            if prev == 0 && next == 0 {
                flowerbed[i] = 1
                count++
            }
        }
    }
    
    return count
}
```

#### **Using Dynamic Programming**
```go
func canPlaceFlowersDP(flowerbed []int, n int) bool {
    if n == 0 {
        return true
    }
    
    dp := make([]int, len(flowerbed))
    
    for i := 0; i < len(flowerbed); i++ {
        if flowerbed[i] == 1 {
            dp[i] = 0
        } else {
            canPlant := true
            
            if i > 0 && flowerbed[i-1] == 1 {
                canPlant = false
            }
            
            if i < len(flowerbed)-1 && flowerbed[i+1] == 1 {
                canPlant = false
            }
            
            if canPlant {
                dp[i] = 1
            } else {
                dp[i] = 0
            }
        }
    }
    
    total := 0
    for i := 0; i < len(dp); i++ {
        if dp[i] == 1 {
            total++
            // Mark adjacent positions as unavailable
            if i > 0 {
                dp[i-1] = 0
            }
            if i < len(dp)-1 {
                dp[i+1] = 0
            }
        }
    }
    
    return total >= n
}
```

### Complexity
- **Time Complexity:** O(n) where n is the length of the flowerbed
- **Space Complexity:** O(1)
