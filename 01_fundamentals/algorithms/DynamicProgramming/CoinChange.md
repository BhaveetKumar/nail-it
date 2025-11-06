---
# Auto-generated front matter
Title: Coinchange
LastUpdated: 2025-11-06T20:45:58.745640
Tags: []
Status: draft
---

# Coin Change

### Problem
You are given an integer array `coins` representing coins of different denominations and an integer `amount` representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return `-1`.

You may assume that you have an infinite number of each kind of coin.

**Example:**
```
Input: coins = [1,3,4], amount = 6
Output: 2
Explanation: 6 = 3 + 3

Input: coins = [2], amount = 3
Output: -1
```

### Golang Solution

```go
func coinChange(coins []int, amount int) int {
    dp := make([]int, amount+1)
    for i := 1; i <= amount; i++ {
        dp[i] = amount + 1 // Initialize with impossible value
    }
    
    for i := 1; i <= amount; i++ {
        for _, coin := range coins {
            if coin <= i {
                dp[i] = min(dp[i], dp[i-coin]+1)
            }
        }
    }
    
    if dp[amount] > amount {
        return -1
    }
    return dp[amount]
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

### Complexity
- **Time Complexity:** O(amount × len(coins))
- **Space Complexity:** O(amount)
