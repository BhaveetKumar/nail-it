pub fn binary_search<T: Ord>(slice: &[T], target: &T) -> Option<usize> {
    let mut lo = 0usize;
    let mut hi = slice.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match slice[mid].cmp(target) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
            std::cmp::Ordering::Equal => return Some(mid),
        }
    }
    None
}

pub fn quicksort<T: Ord>(arr: &mut [T]) {
    if arr.len() <= 1 { return; }
    let len = arr.len();
    let pivot_index = partition(arr);
    quicksort(&mut arr[..pivot_index]);
    quicksort(&mut arr[pivot_index+1..len]);
}

fn partition<T: Ord>(arr: &mut [T]) -> usize {
    let len = arr.len();
    let pivot_index = len - 1;
    let mut store = 0usize;
    for i in 0..pivot_index {
        if arr[i] <= arr[pivot_index] {
            arr.swap(i, store);
            store += 1;
        }
    }
    arr.swap(store, pivot_index);
    store
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_search_basic() {
        let data = [1,2,3,4,5];
        assert_eq!(binary_search(&data, &3), Some(2));
        assert_eq!(binary_search(&data, &6), None);
        assert_eq!(binary_search(&data, &0), None);
    }

    #[test]
    fn quicksort_basic() {
        let mut data = [3,1,4,1,5,9,2,6];
        quicksort(&mut data);
        assert_eq!(data, [1,1,2,3,4,5,6,9]);
    }

    #[test]
    fn quicksort_empty_and_single() {
        let mut empty: [i32; 0] = [];
        quicksort(&mut empty);
        assert_eq!(empty, []);

        let mut single = [42];
        quicksort(&mut single);
        assert_eq!(single, [42]);
    }
}

#[cfg(test)]
mod prop {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn quicksort_sorts(vec in proptest::collection::vec(any::<i32>(), 0..100)) {
            let mut a = vec.clone();
            quicksort(&mut a);
            prop_assert!(a.windows(2).all(|w| w[0] <= w[1]));
        }
    }
}
