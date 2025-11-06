---
# Auto-generated front matter
Title: Dijkstra
LastUpdated: 2025-11-06T20:45:58.795944
Tags: []
Status: draft
---

# Dijkstra's Algorithm

## Problem Statement

Dijkstra's algorithm is used to find the shortest path from a source vertex to all other vertices in a weighted graph with non-negative edge weights.

**Example:**
```
Graph:
    1 --2-- 2
   /|      /|\
  3 |     1 3 2
 /  |    /  |  \
0 --1-- 3 --4-- 5
   \|      /|\
    2     1 2 1
     \    /  |  \
      4 --1-- 5 --1-- 6

Find shortest path from vertex 0 to all other vertices.
```

## Approach

### Dijkstra's Algorithm
1. Initialize distances to all vertices as infinity, except source (0)
2. Create a priority queue (min-heap) with source vertex
3. While queue is not empty:
   - Extract vertex with minimum distance
   - For each neighbor, if new path is shorter, update distance
   - Add neighbor to queue if distance is updated

**Time Complexity:** O((V + E) log V) - With binary heap
**Space Complexity:** O(V) - For distances and priority queue

## Solution

```javascript
/**
 * Find shortest distances from source to all vertices using Dijkstra's algorithm
 * @param {number} n - Number of vertices
 * @param {number[][]} edges - Array of [u, v, weight] representing edges
 * @param {number} src - Source vertex
 * @return {number[]} - Shortest distances from source to all vertices
 */
function dijkstra(n, edges, src) {
    // Build adjacency list
    const graph = Array(n).fill().map(() => []);
    for (const [u, v, weight] of edges) {
        graph[u].push([v, weight]);
        graph[v].push([u, weight]); // For undirected graph
    }
    
    // Initialize distances
    const distances = Array(n).fill(Infinity);
    distances[src] = 0;
    
    // Priority queue: [distance, vertex]
    const pq = [[0, src]];
    
    while (pq.length > 0) {
        // Extract minimum distance vertex
        const [dist, u] = extractMin(pq);
        
        // Skip if we've already processed this vertex
        if (dist > distances[u]) {
            continue;
        }
        
        // Relax edges
        for (const [v, weight] of graph[u]) {
            const newDist = distances[u] + weight;
            
            if (newDist < distances[v]) {
                distances[v] = newDist;
                pq.push([newDist, v]);
            }
        }
    }
    
    return distances;
}

// Helper function to extract minimum from priority queue
function extractMin(pq) {
    let minIndex = 0;
    for (let i = 1; i < pq.length; i++) {
        if (pq[i][0] < pq[minIndex][0]) {
            minIndex = i;
        }
    }
    return pq.splice(minIndex, 1)[0];
}

// Alternative implementation using a proper min-heap
class MinHeap {
    constructor() {
        this.heap = [];
    }
    
    push(item) {
        this.heap.push(item);
        this.heapifyUp(this.heap.length - 1);
    }
    
    pop() {
        if (this.heap.length === 0) return null;
        if (this.heap.length === 1) return this.heap.pop();
        
        const min = this.heap[0];
        this.heap[0] = this.heap.pop();
        this.heapifyDown(0);
        return min;
    }
    
    heapifyUp(index) {
        while (index > 0) {
            const parentIndex = Math.floor((index - 1) / 2);
            if (this.heap[parentIndex][0] <= this.heap[index][0]) break;
            
            [this.heap[parentIndex], this.heap[index]] = [this.heap[index], this.heap[parentIndex]];
            index = parentIndex;
        }
    }
    
    heapifyDown(index) {
        while (true) {
            let smallest = index;
            const left = 2 * index + 1;
            const right = 2 * index + 2;
            
            if (left < this.heap.length && this.heap[left][0] < this.heap[smallest][0]) {
                smallest = left;
            }
            
            if (right < this.heap.length && this.heap[right][0] < this.heap[smallest][0]) {
                smallest = right;
            }
            
            if (smallest === index) break;
            
            [this.heap[index], this.heap[smallest]] = [this.heap[smallest], this.heap[index]];
            index = smallest;
        }
    }
    
    isEmpty() {
        return this.heap.length === 0;
    }
}

function dijkstraWithHeap(n, edges, src) {
    // Build adjacency list
    const graph = Array(n).fill().map(() => []);
    for (const [u, v, weight] of edges) {
        graph[u].push([v, weight]);
        graph[v].push([u, weight]);
    }
    
    // Initialize distances
    const distances = Array(n).fill(Infinity);
    distances[src] = 0;
    
    // Priority queue
    const pq = new MinHeap();
    pq.push([0, src]);
    
    while (!pq.isEmpty()) {
        const [dist, u] = pq.pop();
        
        if (dist > distances[u]) {
            continue;
        }
        
        for (const [v, weight] of graph[u]) {
            const newDist = distances[u] + weight;
            
            if (newDist < distances[v]) {
                distances[v] = newDist;
                pq.push([newDist, v]);
            }
        }
    }
    
    return distances;
}
```

## Dry Run

**Input:** n = 4, edges = [[0,1,4], [0,2,1], [1,2,2], [1,3,5], [2,3,1]], src = 0

```
Graph:
    0 --4-- 1
   /|      /|
  1 |     2 5
 /  |    /  |
2 --2-- 3 --1-- 3

Initial: distances = [0, ∞, ∞, ∞], pq = [[0, 0]]

Step 1: Extract [0, 0]
        distances = [0, ∞, ∞, ∞]
        Neighbors of 0: (1,4), (2,1)
        - v=1: newDist = 0+4 = 4 < ∞, update distances[1] = 4, add [4,1] to pq
        - v=2: newDist = 0+1 = 1 < ∞, update distances[2] = 1, add [1,2] to pq
        pq = [[1,2], [4,1]]

Step 2: Extract [1, 2]
        distances = [0, 4, 1, ∞]
        Neighbors of 2: (0,1), (1,2), (3,1)
        - v=0: newDist = 1+1 = 2 > 0, skip
        - v=1: newDist = 1+2 = 3 < 4, update distances[1] = 3, add [3,1] to pq
        - v=3: newDist = 1+1 = 2 < ∞, update distances[3] = 2, add [2,3] to pq
        pq = [[2,3], [3,1], [4,1]]

Step 3: Extract [2, 3]
        distances = [0, 3, 1, 2]
        Neighbors of 3: (1,5), (2,1)
        - v=1: newDist = 2+5 = 7 > 3, skip
        - v=2: newDist = 2+1 = 3 > 1, skip
        pq = [[3,1], [4,1]]

Step 4: Extract [3, 1]
        distances = [0, 3, 1, 2]
        Neighbors of 1: (0,4), (2,2), (3,5)
        - v=0: newDist = 3+4 = 7 > 0, skip
        - v=2: newDist = 3+2 = 5 > 1, skip
        - v=3: newDist = 3+5 = 8 > 2, skip
        pq = [[4,1]]

Step 5: Extract [4, 1]
        dist = 4 > distances[1] = 3, skip
        pq = []

Result: distances = [0, 3, 1, 2]
```

## Complexity Analysis

- **Time Complexity:** O((V + E) log V) - With binary heap
- **Space Complexity:** O(V) - For distances and priority queue

## Alternative Solutions

### Using Set for Visited Vertices
```javascript
function dijkstraWithSet(n, edges, src) {
    const graph = Array(n).fill().map(() => []);
    for (const [u, v, weight] of edges) {
        graph[u].push([v, weight]);
        graph[v].push([u, weight]);
    }
    
    const distances = Array(n).fill(Infinity);
    distances[src] = 0;
    
    const visited = new Set();
    const pq = new MinHeap();
    pq.push([0, src]);
    
    while (!pq.isEmpty()) {
        const [dist, u] = pq.pop();
        
        if (visited.has(u)) continue;
        visited.add(u);
        
        for (const [v, weight] of graph[u]) {
            if (!visited.has(v)) {
                const newDist = distances[u] + weight;
                if (newDist < distances[v]) {
                    distances[v] = newDist;
                    pq.push([newDist, v]);
                }
            }
        }
    }
    
    return distances;
}
```

### Finding Shortest Path (Not Just Distance)
```javascript
function dijkstraWithPath(n, edges, src, dest) {
    const graph = Array(n).fill().map(() => []);
    for (const [u, v, weight] of edges) {
        graph[u].push([v, weight]);
        graph[v].push([u, weight]);
    }
    
    const distances = Array(n).fill(Infinity);
    const parents = Array(n).fill(-1);
    distances[src] = 0;
    
    const pq = new MinHeap();
    pq.push([0, src]);
    
    while (!pq.isEmpty()) {
        const [dist, u] = pq.pop();
        
        if (dist > distances[u]) continue;
        
        for (const [v, weight] of graph[u]) {
            const newDist = distances[u] + weight;
            if (newDist < distances[v]) {
                distances[v] = newDist;
                parents[v] = u;
                pq.push([newDist, v]);
            }
        }
    }
    
    // Reconstruct path
    const path = [];
    let current = dest;
    while (current !== -1) {
        path.unshift(current);
        current = parents[current];
    }
    
    return {
        distance: distances[dest],
        path: path
    };
}
```

## Test Cases

```javascript
// Test cases
const edges1 = [[0,1,4], [0,2,1], [1,2,2], [1,3,5], [2,3,1]];
console.log(dijkstra(4, edges1, 0)); // [0, 3, 1, 2]

const edges2 = [[0,1,1], [0,2,3], [1,2,1], [1,3,2], [2,3,1]];
console.log(dijkstra(4, edges2, 0)); // [0, 1, 2, 3]

const edges3 = [[0,1,2], [0,2,4], [1,2,1], [1,3,7], [2,3,3]];
console.log(dijkstra(4, edges3, 0)); // [0, 2, 3, 6]

// Path finding
const result = dijkstraWithPath(4, edges1, 0, 3);
console.log(result); // { distance: 2, path: [0, 2, 3] }
```

## Key Insights

1. **Greedy Algorithm**: Always chooses the vertex with minimum distance
2. **Non-negative Weights**: Only works with non-negative edge weights
3. **Single Source**: Finds shortest paths from one source to all vertices
4. **Priority Queue**: Essential for efficient implementation
5. **Relaxation**: Process of updating distances when shorter path found

## Related Problems

- [Network Delay Time](../../../algorithms/Graphs/NetworkDelayTime.md) - Dijkstra with time constraints
- [Cheapest Flights Within K Stops](CheapestFlights.md) - Dijkstra with stop limit
- [Path With Minimum Effort](../../../algorithms/Graphs/PathWithMinimumEffort.md) - Dijkstra on 2D grid
- [Bellman-Ford Algorithm](BellmanFord.md) - For negative weights
- [Floyd-Warshall Algorithm](FloydWarshall.md) - All pairs shortest paths
