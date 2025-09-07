# Sorting Pattern

> **Master sorting algorithms and their applications with Go implementations**

## 📋 Problems

### **Basic Sorting**
- [Quick Sort](./QuickSort.md) - Divide and conquer sorting
- [Merge Sort](./MergeSort.md) - Stable divide and conquer sorting
- [Heap Sort](./HeapSort.md) - In-place sorting using heap
- [Insertion Sort](./InsertionSort.md) - Simple comparison-based sorting
- [Selection Sort](./SelectionSort.md) - Simple in-place sorting

### **Advanced Sorting**
- [Counting Sort](./CountingSort.md) - Non-comparison based sorting
- [Radix Sort](./RadixSort.md) - Digit-based sorting
- [Bucket Sort](./BucketSort.md) - Distribution-based sorting
- [Shell Sort](./ShellSort.md) - Improved insertion sort
- [Tim Sort](./TimSort.md) - Hybrid stable sorting

### **Sorting Applications**
- [Sort Colors](./SortColors.md) - Dutch National Flag problem
- [Wiggle Sort](./WiggleSort.md) - Alternating sort
- [H-Index](./HIndex.md) - Citation-based sorting
- [Largest Number](./LargestNumber.md) - Custom comparison sorting
- [Meeting Rooms](./MeetingRooms.md) - Interval sorting

---

## 🎯 Key Concepts

### **Sorting Algorithms**
- **Comparison-based**: Quick Sort, Merge Sort, Heap Sort
- **Non-comparison**: Counting Sort, Radix Sort, Bucket Sort
- **Stable**: Merge Sort, Insertion Sort, Tim Sort
- **In-place**: Quick Sort, Heap Sort, Insertion Sort

### **Time Complexity**
- **O(n log n)**: Quick Sort, Merge Sort, Heap Sort
- **O(n²)**: Insertion Sort, Selection Sort, Bubble Sort
- **O(n)**: Counting Sort, Radix Sort, Bucket Sort
- **O(n log n) average**: Quick Sort

---

## 🛠️ Go-Specific Tips

### **Built-in Sorting**
```go
import "sort"

// Sort integers
nums := []int{3, 1, 4, 1, 5}
sort.Ints(nums)

// Sort strings
strs := []string{"banana", "apple", "cherry"}
sort.Strings(strs)

// Custom sorting
type Person struct {
    Name string
    Age  int
}

people := []Person{{"Alice", 30}, {"Bob", 25}}
sort.Slice(people, func(i, j int) bool {
    return people[i].Age < people[j].Age
})
```

### **Quick Sort Implementation**
```go
func quickSort(nums []int) {
    if len(nums) <= 1 {
        return
    }
    
    pivot := partition(nums)
    quickSort(nums[:pivot])
    quickSort(nums[pivot+1:])
}

func partition(nums []int) int {
    pivot := nums[len(nums)-1]
    i := 0
    
    for j := 0; j < len(nums)-1; j++ {
        if nums[j] <= pivot {
            nums[i], nums[j] = nums[j], nums[i]
            i++
        }
    }
    
    nums[i], nums[len(nums)-1] = nums[len(nums)-1], nums[i]
    return i
}
```

---

## 🎯 Interview Tips

### **How to Identify Sorting Problems**
1. **Ordering Requirements**: Need elements in specific order
2. **Comparison Problems**: Need to compare elements
3. **Search Optimization**: Sort to enable binary search
4. **Duplicate Handling**: Sort to group duplicates

### **Common Sorting Problem Patterns**
- **Array Sorting**: Sort entire array
- **Partial Sorting**: Sort first k elements
- **Custom Comparison**: Sort with custom criteria
- **Stable Sorting**: Maintain relative order of equal elements

### **Optimization Tips**
- **Use Built-in Sort**: Leverage Go's optimized sort package
- **Choose Right Algorithm**: Consider stability, space, and time requirements
- **Custom Comparison**: Use sort.Slice for complex sorting criteria
- **Avoid Unnecessary Sorting**: Only sort when required
