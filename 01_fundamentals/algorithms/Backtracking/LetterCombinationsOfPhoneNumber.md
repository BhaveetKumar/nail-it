# Letter Combinations of a Phone Number

### Problem
Given a string containing digits from `2-9` inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

```
2: "abc"
3: "def"
4: "ghi"
5: "jkl"
6: "mno"
7: "pqrs"
8: "tuv"
9: "wxyz"
```

**Example:**
```
Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]

Input: digits = ""
Output: []

Input: digits = "2"
Output: ["a","b","c"]
```

### Golang Solution

```go
func letterCombinations(digits string) []string {
    if len(digits) == 0 {
        return []string{}
    }
    
    digitToLetters := map[byte]string{
        '2': "abc", '3': "def", '4': "ghi", '5': "jkl",
        '6': "mno", '7': "pqrs", '8': "tuv", '9': "wxyz",
    }
    
    var result []string
    var backtrack func(int, string)
    
    backtrack = func(index int, current string) {
        if index == len(digits) {
            result = append(result, current)
            return
        }
        
        digit := digits[index]
        letters := digitToLetters[digit]
        
        for _, letter := range letters {
            backtrack(index+1, current+string(letter))
        }
    }
    
    backtrack(0, "")
    return result
}
```

### Alternative Solutions

#### **Iterative Approach**
```go
func letterCombinationsIterative(digits string) []string {
    if len(digits) == 0 {
        return []string{}
    }
    
    digitToLetters := map[byte]string{
        '2': "abc", '3': "def", '4': "ghi", '5': "jkl",
        '6': "mno", '7': "pqrs", '8': "tuv", '9': "wxyz",
    }
    
    result := []string{""}
    
    for _, digit := range digits {
        letters := digitToLetters[digit]
        newResult := []string{}
        
        for _, combination := range result {
            for _, letter := range letters {
                newResult = append(newResult, combination+string(letter))
            }
        }
        
        result = newResult
    }
    
    return result
}
```

#### **Using Queue**
```go
func letterCombinationsQueue(digits string) []string {
    if len(digits) == 0 {
        return []string{}
    }
    
    digitToLetters := map[byte]string{
        '2': "abc", '3': "def", '4': "ghi", '5': "jkl",
        '6': "mno", '7': "pqrs", '8': "tuv", '9': "wxyz",
    }
    
    queue := []string{""}
    
    for _, digit := range digits {
        letters := digitToLetters[digit]
        levelSize := len(queue)
        
        for i := 0; i < levelSize; i++ {
            current := queue[0]
            queue = queue[1:]
            
            for _, letter := range letters {
                queue = append(queue, current+string(letter))
            }
        }
    }
    
    return queue
}
```

### Complexity
- **Time Complexity:** O(4^n × n) where n is length of digits
- **Space Complexity:** O(4^n × n)
