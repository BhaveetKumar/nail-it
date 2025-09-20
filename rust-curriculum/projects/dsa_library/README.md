# DSA Library

Algorithms included:

- `binary_search(&[T], &T) -> Option<usize>`
- `quicksort(&mut [T])`
- `bfs(adj: &HashMap<usize, Vec<usize>>, start) -> Vec<usize>`
- `dfs(adj: &HashMap<usize, Vec<usize>>, start) -> Vec<usize>`
- `dijkstra(adj: &HashMap<usize, Vec<(usize,i64)>>, start) -> HashMap<usize, i64>`

Examples:

```rust
use dsa_library::{binary_search, quicksort};
let data = [1,2,3,4,5];
assert_eq!(binary_search(&data, &3), Some(2));
let mut v = vec![3,1,4];
quicksort(&mut v);
assert_eq!(v, vec![1,3,4]);
```

Benches:

```sh
cargo bench -p dsa_library
```

## Milestones

Implement classic DS & algorithms with safe paths and optional unsafe optimizations (feature-gated), plus Criterion benchmarks.

- Core structures: vector, list, stack, queue, deque, heap.
- Trees: BST, AVL, Red-Black.
- Graphs: BFS, DFS, Dijkstra, A*.
- Sorting & searching.
- Benchmarks with Criterion.
- Benchmarks with Criterion.


---

## AUTO-GENERATED: Starter Content

<!-- AUTO-GENERATED - REVIEW REQUIRED -->

This section seeds the document with a short introduction, learning objectives, and related links to code samples.

**Learning objectives:**
- Understand the core concepts.
- See practical code examples.

**Related files:**

Please replace this auto-generated section with curated content.


---

## AUTO-GENERATED: Starter Content

<!-- AUTO-GENERATED - REVIEW REQUIRED -->

This section seeds the document with a short introduction, learning objectives, and related links to code samples.

**Learning objectives:**
- Understand the core concepts.
- See practical code examples.

**Related files:**

Please replace this auto-generated section with curated content.
