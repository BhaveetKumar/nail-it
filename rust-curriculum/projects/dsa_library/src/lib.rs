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

use std::collections::{VecDeque, BinaryHeap, HashMap};

pub fn bfs(adj: &HashMap<usize, Vec<usize>>, start: usize) -> Vec<usize> {
    let mut visited = vec![];
    let mut seen = std::collections::HashSet::new();
    let mut q = VecDeque::new();
    q.push_back(start);
    seen.insert(start);
    while let Some(u) = q.pop_front() {
        visited.push(u);
        if let Some(neis) = adj.get(&u) {
            for &v in neis {
                if !seen.contains(&v) { seen.insert(v); q.push_back(v); }
            }
        }
    }
    visited
}

pub fn dfs(adj: &HashMap<usize, Vec<usize>>, start: usize) -> Vec<usize> {
    fn rec(adj: &HashMap<usize, Vec<usize>>, u: usize, seen: &mut std::collections::HashSet<usize>, out: &mut Vec<usize>) {
        seen.insert(u);
        out.push(u);
        if let Some(neis) = adj.get(&u) {
            for &v in neis {
                if !seen.contains(&v) { rec(adj, v, seen, out); }
            }
        }
    }
    let mut out = vec![];
    let mut seen = std::collections::HashSet::new();
    rec(adj, start, &mut seen, &mut out);
    out
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct State { cost: i64, node: usize }

impl Ord for State { fn cmp(&self, other: &Self) -> std::cmp::Ordering { other.cost.cmp(&self.cost).then(self.node.cmp(&other.node)) } }
impl PartialOrd for State { fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) } }

pub fn dijkstra(adj: &HashMap<usize, Vec<(usize, i64)>>, start: usize) -> HashMap<usize, i64> {
    let mut dist: HashMap<usize, i64> = HashMap::new();
    let mut heap = BinaryHeap::new();
    dist.insert(start, 0);
    heap.push(State { cost: 0, node: start });
    while let Some(State { cost, node }) = heap.pop() {
        if let Some(&d) = dist.get(&node) { if cost > d { continue; } }
        if let Some(neis) = adj.get(&node) {
            for &(v, w) in neis {
                let next = cost + w;
                if dist.get(&v).map_or(true, |&cur| next < cur) {
                    dist.insert(v, next);
                    heap.push(State { cost: next, node: v });
                }
            }
        }
    }
    dist
}

#[cfg(test)]
mod graph_tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn bfs_dfs_basic() {
        let mut g: HashMap<usize, Vec<usize>> = HashMap::new();
        g.insert(0, vec![1,2]);
        g.insert(1, vec![3]);
        g.insert(2, vec![]);
        g.insert(3, vec![]);
        assert_eq!(bfs(&g, 0), vec![0,1,2,3]);
        assert_eq!(dfs(&g, 0), vec![0,1,3,2]);
    }

    #[test]
    fn dijkstra_basic() {
        let mut g: HashMap<usize, Vec<(usize, i64)>> = HashMap::new();
        g.insert(0, vec![(1, 2), (2, 5)]);
        g.insert(1, vec![(2, 1)]);
        g.insert(2, vec![]);
        let d = dijkstra(&g, 0);
        assert_eq!(d.get(&0).copied(), Some(0));
        assert_eq!(d.get(&1).copied(), Some(2));
        assert_eq!(d.get(&2).copied(), Some(3));
    }
}
