---
# Auto-generated front matter
Title: Lemonadechange
LastUpdated: 2025-11-06T20:45:58.728398
Tags: []
Status: draft
---

# Lemonade Change

### Problem
At a lemonade stand, each lemonade costs `$5`. Customers are standing in a queue to buy from you and order one at a time (in the order specified by bills). Each customer will only buy one lemonade and pay with either a `$5`, `$10`, or `$20` bill. You must provide the correct change to each customer so that the net transaction is that the customer pays `$5`.

Note that you don't have any change in hand at first.

Given an integer array `bills` where `bills[i]` is the bill the `ith` customer pays, return `true` if you can provide every customer with the correct change, or `false` otherwise.

**Example:**
```
Input: bills = [5,5,5,10,20]
Output: true
Explanation: 
From the first 3 customers, we collect three $5 bills in order.
From the fourth customer, we collect a $10 bill and give back a $5.
From the fifth customer, we collect a $20 bill and give back a $10 and a $5.
Since all customers got correct change, we output true.

Input: bills = [5,5,10,10,20]
Output: false
Explanation: 
From the first two customers in order, we collect two $5 bills.
For the next two customers in order, we collect a $10 bill and give back a $5 bill.
For the last customer, we cannot give the change of $15 back because we only have two $10 bills.
Since not every customer received the correct change, the answer is false.
```

### Golang Solution

```go
func lemonadeChange(bills []int) bool {
    five, ten := 0, 0
    
    for _, bill := range bills {
        switch bill {
        case 5:
            five++
        case 10:
            if five == 0 {
                return false
            }
            five--
            ten++
        case 20:
            if ten > 0 && five > 0 {
                ten--
                five--
            } else if five >= 3 {
                five -= 3
            } else {
                return false
            }
        }
    }
    
    return true
}
```

### Alternative Solutions

#### **Using Array**
```go
func lemonadeChangeArray(bills []int) bool {
    change := make([]int, 3) // [5, 10, 20] counts
    
    for _, bill := range bills {
        change[bill/5-1]++ // Convert to index
        
        if bill == 10 {
            if change[0] == 0 {
                return false
            }
            change[0]--
        } else if bill == 20 {
            if change[1] > 0 && change[0] > 0 {
                change[1]--
                change[0]--
            } else if change[0] >= 3 {
                change[0] -= 3
            } else {
                return false
            }
        }
    }
    
    return true
}
```

#### **Using Map**
```go
func lemonadeChangeMap(bills []int) bool {
    change := make(map[int]int)
    
    for _, bill := range bills {
        change[bill]++
        
        if bill == 10 {
            if change[5] == 0 {
                return false
            }
            change[5]--
        } else if bill == 20 {
            if change[10] > 0 && change[5] > 0 {
                change[10]--
                change[5]--
            } else if change[5] >= 3 {
                change[5] -= 3
            } else {
                return false
            }
        }
    }
    
    return true
}
```

#### **Detailed Logging**
```go
func lemonadeChangeDetailed(bills []int) bool {
    five, ten := 0, 0
    
    for i, bill := range bills {
        fmt.Printf("Customer %d pays $%d\n", i+1, bill)
        
        switch bill {
        case 5:
            five++
            fmt.Printf("  Received $5, total $5 bills: %d\n", five)
        case 10:
            if five == 0 {
                fmt.Printf("  Cannot give change for $10, no $5 bills\n")
                return false
            }
            five--
            ten++
            fmt.Printf("  Gave $5 change, remaining $5: %d, $10: %d\n", five, ten)
        case 20:
            if ten > 0 && five > 0 {
                ten--
                five--
                fmt.Printf("  Gave $10+$5 change, remaining $5: %d, $10: %d\n", five, ten)
            } else if five >= 3 {
                five -= 3
                fmt.Printf("  Gave 3x$5 change, remaining $5: %d\n", five)
            } else {
                fmt.Printf("  Cannot give change for $20\n")
                return false
            }
        }
    }
    
    return true
}
```

#### **Return Change Details**
```go
type ChangeResult struct {
    CanGiveChange bool
    ChangeGiven   []int
    RemainingBills map[int]int
}

func lemonadeChangeWithDetails(bills []int) ChangeResult {
    five, ten := 0, 0
    var changeGiven []int
    
    for _, bill := range bills {
        switch bill {
        case 5:
            five++
        case 10:
            if five == 0 {
                return ChangeResult{false, changeGiven, map[int]int{5: five, 10: ten}}
            }
            five--
            ten++
            changeGiven = append(changeGiven, 5)
        case 20:
            if ten > 0 && five > 0 {
                ten--
                five--
                changeGiven = append(changeGiven, 10, 5)
            } else if five >= 3 {
                five -= 3
                changeGiven = append(changeGiven, 5, 5, 5)
            } else {
                return ChangeResult{false, changeGiven, map[int]int{5: five, 10: ten}}
            }
        }
    }
    
    return ChangeResult{true, changeGiven, map[int]int{5: five, 10: ten}}
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
