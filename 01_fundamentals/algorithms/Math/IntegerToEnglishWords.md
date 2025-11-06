---
# Auto-generated front matter
Title: Integertoenglishwords
LastUpdated: 2025-11-06T20:45:58.714559
Tags: []
Status: draft
---

# Integer to English Words

### Problem
Convert a non-negative integer `num` to its English words representation.

**Example:**
```
Input: num = 123
Output: "One Hundred Twenty Three"

Input: num = 12345
Output: "Twelve Thousand Three Hundred Forty Five"

Input: num = 1234567
Output: "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"
```

### Golang Solution

```go
func numberToWords(num int) string {
    if num == 0 {
        return "Zero"
    }
    
    ones := []string{"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"}
    teens := []string{"Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"}
    tens := []string{"", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"}
    thousands := []string{"", "Thousand", "Million", "Billion"}
    
    var result []string
    
    for i := 0; num > 0; i++ {
        if num%1000 != 0 {
            result = append([]string{helper(num%1000, ones, teens, tens)}, result...)
            if i > 0 {
                result = append([]string{thousands[i]}, result...)
            }
        }
        num /= 1000
    }
    
    return strings.Join(result, " ")
}

func helper(num int, ones, teens, tens []string) string {
    if num == 0 {
        return ""
    }
    
    var result []string
    
    if num >= 100 {
        result = append(result, ones[num/100])
        result = append(result, "Hundred")
        num %= 100
    }
    
    if num >= 20 {
        result = append(result, tens[num/10])
        num %= 10
    } else if num >= 10 {
        result = append(result, teens[num-10])
        return strings.Join(result, " ")
    }
    
    if num > 0 {
        result = append(result, ones[num])
    }
    
    return strings.Join(result, " ")
}
```

### Alternative Solutions

#### **Recursive Approach**
```go
func numberToWordsRecursive(num int) string {
    if num == 0 {
        return "Zero"
    }
    
    ones := []string{"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"}
    teens := []string{"Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"}
    tens := []string{"", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"}
    
    var convert func(int) string
    convert = func(n int) string {
        if n == 0 {
            return ""
        }
        
        if n < 10 {
            return ones[n]
        } else if n < 20 {
            return teens[n-10]
        } else if n < 100 {
            return strings.TrimSpace(tens[n/10] + " " + convert(n%10))
        } else if n < 1000 {
            return strings.TrimSpace(ones[n/100] + " Hundred " + convert(n%100))
        } else if n < 1000000 {
            return strings.TrimSpace(convert(n/1000) + " Thousand " + convert(n%1000))
        } else if n < 1000000000 {
            return strings.TrimSpace(convert(n/1000000) + " Million " + convert(n%1000000))
        } else {
            return strings.TrimSpace(convert(n/1000000000) + " Billion " + convert(n%1000000000))
        }
    }
    
    return convert(num)
}
```

### Complexity
- **Time Complexity:** O(log n)
- **Space Complexity:** O(1)
