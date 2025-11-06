---
# Auto-generated front matter
Title: Graphalgorithms
LastUpdated: 2025-11-06T20:45:58.794892
Tags: []
Status: draft
---

# 🕸️ Graph Algorithms - Complete Implementation

## Problem Statement

Implement comprehensive graph algorithms including traversal, shortest path, topological sort, and cycle detection.

## Graph Representation

### 1. Adjacency List

```javascript
class Graph {
  constructor(vertices) {
    this.vertices = vertices;
    this.adjList = new Map();
    
    // Initialize adjacency list
    for (let i = 0; i < vertices; i++) {
      this.adjList.set(i, []);
    }
  }
  
  // Add edge (undirected)
  addEdge(u, v, weight = 1) {
    this.adjList.get(u).push({ vertex: v, weight });
    this.adjList.get(v).push({ vertex: u, weight });
  }
  
  // Add directed edge
  addDirectedEdge(u, v, weight = 1) {
    this.adjList.get(u).push({ vertex: v, weight });
  }
  
  // Get neighbors
  getNeighbors(vertex) {
    return this.adjList.get(vertex) || [];
  }
}
```

### 2. Adjacency Matrix

```javascript
class GraphMatrix {
  constructor(vertices) {
    this.vertices = vertices;
    this.matrix = Array(vertices).fill().map(() => Array(vertices).fill(0));
  }
  
  addEdge(u, v, weight = 1) {
    this.matrix[u][v] = weight;
    this.matrix[v][u] = weight; // For undirected graph
  }
  
  addDirectedEdge(u, v, weight = 1) {
    this.matrix[u][v] = weight;
  }
  
  hasEdge(u, v) {
    return this.matrix[u][v] !== 0;
  }
}
```

## Graph Traversal

### 1. Depth-First Search (DFS)

```javascript
/**
 * DFS recursive implementation
 * @param {Graph} graph
 * @param {number} start
 * @return {number[]}
 */
function dfsRecursive(graph, start) {
  const visited = new Set();
  const result = [];
  
  function dfs(vertex) {
    visited.add(vertex);
    result.push(vertex);
    
    const neighbors = graph.getNeighbors(vertex);
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor.vertex)) {
        dfs(neighbor.vertex);
      }
    }
  }
  
  dfs(start);
  return result;
}

/**
 * DFS iterative implementation
 * @param {Graph} graph
 * @param {number} start
 * @return {number[]}
 */
function dfsIterative(graph, start) {
  const visited = new Set();
  const result = [];
  const stack = [start];
  
  while (stack.length > 0) {
    const vertex = stack.pop();
    
    if (!visited.has(vertex)) {
      visited.add(vertex);
      result.push(vertex);
      
      const neighbors = graph.getNeighbors(vertex);
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor.vertex)) {
          stack.push(neighbor.vertex);
        }
      }
    }
  }
  
  return result;
}
```

### 2. Breadth-First Search (BFS)

```javascript
/**
 * BFS implementation
 * @param {Graph} graph
 * @param {number} start
 * @return {number[]}
 */
function bfs(graph, start) {
  const visited = new Set();
  const result = [];
  const queue = [start];
  
  visited.add(start);
  
  while (queue.length > 0) {
    const vertex = queue.shift();
    result.push(vertex);
    
    const neighbors = graph.getNeighbors(vertex);
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor.vertex)) {
        visited.add(neighbor.vertex);
        queue.push(neighbor.vertex);
      }
    }
  }
  
  return result;
}

/**
 * BFS with level information
 * @param {Graph} graph
 * @param {number} start
 * @return {number[][]}
 */
function bfsWithLevels(graph, start) {
  const visited = new Set();
  const result = [];
  const queue = [{ vertex: start, level: 0 }];
  
  visited.add(start);
  
  while (queue.length > 0) {
    const { vertex, level } = queue.shift();
    
    if (!result[level]) {
      result[level] = [];
    }
    result[level].push(vertex);
    
    const neighbors = graph.getNeighbors(vertex);
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor.vertex)) {
        visited.add(neighbor.vertex);
        queue.push({ vertex: neighbor.vertex, level: level + 1 });
      }
    }
  }
  
  return result;
}
```

## Shortest Path Algorithms

### 1. Dijkstra's Algorithm

```javascript
/**
 * Dijkstra's algorithm for shortest path
 * @param {Graph} graph
 * @param {number} start
 * @return {Object}
 */
function dijkstra(graph, start) {
  const distances = new Array(graph.vertices).fill(Infinity);
  const previous = new Array(graph.vertices).fill(null);
  const visited = new Set();
  
  distances[start] = 0;
  
  // Priority queue (min heap)
  const pq = [{ vertex: start, distance: 0 }];
  
  while (pq.length > 0) {
    // Sort by distance (simple implementation)
    pq.sort((a, b) => a.distance - b.distance);
    const { vertex, distance } = pq.shift();
    
    if (visited.has(vertex)) continue;
    visited.add(vertex);
    
    const neighbors = graph.getNeighbors(vertex);
    for (const neighbor of neighbors) {
      const newDistance = distance + neighbor.weight;
      
      if (newDistance < distances[neighbor.vertex]) {
        distances[neighbor.vertex] = newDistance;
        previous[neighbor.vertex] = vertex;
        pq.push({ vertex: neighbor.vertex, distance: newDistance });
      }
    }
  }
  
  return { distances, previous };
}

/**
 * Get shortest path from start to end
 * @param {Object} dijkstraResult
 * @param {number} start
 * @param {number} end
 * @return {number[]}
 */
function getShortestPath(dijkstraResult, start, end) {
  const { distances, previous } = dijkstraResult;
  const path = [];
  
  if (distances[end] === Infinity) return null;
  
  let current = end;
  while (current !== null) {
    path.unshift(current);
    current = previous[current];
  }
  
  return path;
}
```

### 2. Bellman-Ford Algorithm

```javascript
/**
 * Bellman-Ford algorithm for shortest path with negative weights
 * @param {Graph} graph
 * @param {number} start
 * @return {Object}
 */
function bellmanFord(graph, start) {
  const distances = new Array(graph.vertices).fill(Infinity);
  const previous = new Array(graph.vertices).fill(null);
  
  distances[start] = 0;
  
  // Relax edges V-1 times
  for (let i = 0; i < graph.vertices - 1; i++) {
    for (let u = 0; u < graph.vertices; u++) {
      const neighbors = graph.getNeighbors(u);
      for (const neighbor of neighbors) {
        const v = neighbor.vertex;
        const weight = neighbor.weight;
        
        if (distances[u] !== Infinity && distances[u] + weight < distances[v]) {
          distances[v] = distances[u] + weight;
          previous[v] = u;
        }
      }
    }
  }
  
  // Check for negative cycles
  for (let u = 0; u < graph.vertices; u++) {
    const neighbors = graph.getNeighbors(u);
    for (const neighbor of neighbors) {
      const v = neighbor.vertex;
      const weight = neighbor.weight;
      
      if (distances[u] !== Infinity && distances[u] + weight < distances[v]) {
        return { hasNegativeCycle: true };
      }
    }
  }
  
  return { distances, previous, hasNegativeCycle: false };
}
```

### 3. Floyd-Warshall Algorithm

```javascript
/**
 * Floyd-Warshall algorithm for all-pairs shortest path
 * @param {number[][]} graph
 * @return {number[][]}
 */
function floydWarshall(graph) {
  const n = graph.length;
  const dist = graph.map(row => [...row]);
  
  // Initialize distances
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j && dist[i][j] === 0) {
        dist[i][j] = Infinity;
      }
    }
  }
  
  // Floyd-Warshall algorithm
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (dist[i][k] !== Infinity && dist[k][j] !== Infinity) {
          dist[i][j] = Math.min(dist[i][j], dist[i][k] + dist[k][j]);
        }
      }
    }
  }
  
  return dist;
}
```

## Topological Sort

```javascript
/**
 * Topological sort using DFS
 * @param {Graph} graph
 * @return {number[]}
 */
function topologicalSort(graph) {
  const visited = new Set();
  const stack = [];
  
  function dfs(vertex) {
    visited.add(vertex);
    
    const neighbors = graph.getNeighbors(vertex);
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor.vertex)) {
        dfs(neighbor.vertex);
      }
    }
    
    stack.push(vertex);
  }
  
  for (let i = 0; i < graph.vertices; i++) {
    if (!visited.has(i)) {
      dfs(i);
    }
  }
  
  return stack.reverse();
}

/**
 * Topological sort using Kahn's algorithm (BFS)
 * @param {Graph} graph
 * @return {number[]}
 */
function topologicalSortKahn(graph) {
  const inDegree = new Array(graph.vertices).fill(0);
  
  // Calculate in-degrees
  for (let i = 0; i < graph.vertices; i++) {
    const neighbors = graph.getNeighbors(i);
    for (const neighbor of neighbors) {
      inDegree[neighbor.vertex]++;
    }
  }
  
  const queue = [];
  const result = [];
  
  // Add vertices with in-degree 0
  for (let i = 0; i < graph.vertices; i++) {
    if (inDegree[i] === 0) {
      queue.push(i);
    }
  }
  
  while (queue.length > 0) {
    const vertex = queue.shift();
    result.push(vertex);
    
    const neighbors = graph.getNeighbors(vertex);
    for (const neighbor of neighbors) {
      inDegree[neighbor.vertex]--;
      if (inDegree[neighbor.vertex] === 0) {
        queue.push(neighbor.vertex);
      }
    }
  }
  
  return result.length === graph.vertices ? result : null;
}
```

## Cycle Detection

### 1. Detect Cycle in Directed Graph

```javascript
/**
 * Detect cycle in directed graph using DFS
 * @param {Graph} graph
 * @return {boolean}
 */
function hasCycleDirected(graph) {
  const visited = new Set();
  const recursionStack = new Set();
  
  function dfs(vertex) {
    visited.add(vertex);
    recursionStack.add(vertex);
    
    const neighbors = graph.getNeighbors(vertex);
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor.vertex)) {
        if (dfs(neighbor.vertex)) return true;
      } else if (recursionStack.has(neighbor.vertex)) {
        return true;
      }
    }
    
    recursionStack.delete(vertex);
    return false;
  }
  
  for (let i = 0; i < graph.vertices; i++) {
    if (!visited.has(i)) {
      if (dfs(i)) return true;
    }
  }
  
  return false;
}
```

### 2. Detect Cycle in Undirected Graph

```javascript
/**
 * Detect cycle in undirected graph using DFS
 * @param {Graph} graph
 * @return {boolean}
 */
function hasCycleUndirected(graph) {
  const visited = new Set();
  
  function dfs(vertex, parent) {
    visited.add(vertex);
    
    const neighbors = graph.getNeighbors(vertex);
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor.vertex)) {
        if (dfs(neighbor.vertex, vertex)) return true;
      } else if (neighbor.vertex !== parent) {
        return true;
      }
    }
    
    return false;
  }
  
  for (let i = 0; i < graph.vertices; i++) {
    if (!visited.has(i)) {
      if (dfs(i, -1)) return true;
    }
  }
  
  return false;
}
```

## Minimum Spanning Tree

### 1. Kruskal's Algorithm

```javascript
/**
 * Kruskal's algorithm for MST
 * @param {Graph} graph
 * @return {Array}
 */
function kruskalMST(graph) {
  const edges = [];
  const mst = [];
  
  // Collect all edges
  for (let i = 0; i < graph.vertices; i++) {
    const neighbors = graph.getNeighbors(i);
    for (const neighbor of neighbors) {
      if (i < neighbor.vertex) { // Avoid duplicates
        edges.push({ u: i, v: neighbor.vertex, weight: neighbor.weight });
      }
    }
  }
  
  // Sort edges by weight
  edges.sort((a, b) => a.weight - b.weight);
  
  // Union-Find data structure
  const parent = Array(graph.vertices).fill().map((_, i) => i);
  
  function find(x) {
    if (parent[x] !== x) {
      parent[x] = find(parent[x]);
    }
    return parent[x];
  }
  
  function union(x, y) {
    const rootX = find(x);
    const rootY = find(y);
    if (rootX !== rootY) {
      parent[rootX] = rootY;
      return true;
    }
    return false;
  }
  
  // Process edges
  for (const edge of edges) {
    if (union(edge.u, edge.v)) {
      mst.push(edge);
      if (mst.length === graph.vertices - 1) break;
    }
  }
  
  return mst;
}
```

### 2. Prim's Algorithm

```javascript
/**
 * Prim's algorithm for MST
 * @param {Graph} graph
 * @return {Array}
 */
function primMST(graph) {
  const mst = [];
  const visited = new Set();
  const pq = []; // Priority queue
  
  // Start with vertex 0
  visited.add(0);
  const neighbors = graph.getNeighbors(0);
  for (const neighbor of neighbors) {
    pq.push({ u: 0, v: neighbor.vertex, weight: neighbor.weight });
  }
  
  while (pq.length > 0 && mst.length < graph.vertices - 1) {
    // Sort by weight (simple implementation)
    pq.sort((a, b) => a.weight - b.weight);
    const edge = pq.shift();
    
    if (!visited.has(edge.v)) {
      visited.add(edge.v);
      mst.push(edge);
      
      const neighbors = graph.getNeighbors(edge.v);
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor.vertex)) {
          pq.push({ u: edge.v, v: neighbor.vertex, weight: neighbor.weight });
        }
      }
    }
  }
  
  return mst;
}
```

## Test Cases

```javascript
// Test cases
console.log("=== Graph Algorithms Test ===");

// Create graph
const graph = new Graph(6);
graph.addDirectedEdge(0, 1, 2);
graph.addDirectedEdge(0, 2, 4);
graph.addDirectedEdge(1, 2, 1);
graph.addDirectedEdge(1, 3, 7);
graph.addDirectedEdge(2, 4, 3);
graph.addDirectedEdge(3, 4, 2);
graph.addDirectedEdge(3, 5, 1);
graph.addDirectedEdge(4, 5, 5);

console.log("Graph created with 6 vertices");

// Traversal tests
console.log("\n=== Traversal Tests ===");
console.log("DFS (recursive) from 0:", dfsRecursive(graph, 0));
console.log("DFS (iterative) from 0:", dfsIterative(graph, 0));
console.log("BFS from 0:", bfs(graph, 0));
console.log("BFS with levels:", bfsWithLevels(graph, 0));

// Shortest path tests
console.log("\n=== Shortest Path Tests ===");
const dijkstraResult = dijkstra(graph, 0);
console.log("Dijkstra distances:", dijkstraResult.distances);
console.log("Shortest path 0->5:", getShortestPath(dijkstraResult, 0, 5));

// Topological sort
console.log("\n=== Topological Sort ===");
const topoGraph = new Graph(6);
topoGraph.addDirectedEdge(5, 2, 1);
topoGraph.addDirectedEdge(5, 0, 1);
topoGraph.addDirectedEdge(4, 0, 1);
topoGraph.addDirectedEdge(4, 1, 1);
topoGraph.addDirectedEdge(2, 3, 1);
topoGraph.addDirectedEdge(3, 1, 1);

console.log("Topological sort (DFS):", topologicalSort(topoGraph));
console.log("Topological sort (Kahn):", topologicalSortKahn(topoGraph));

// Cycle detection
console.log("\n=== Cycle Detection ===");
const cycleGraph = new Graph(4);
cycleGraph.addDirectedEdge(0, 1, 1);
cycleGraph.addDirectedEdge(1, 2, 1);
cycleGraph.addDirectedEdge(2, 3, 1);
cycleGraph.addDirectedEdge(3, 1, 1); // Creates cycle

console.log("Has cycle (directed):", hasCycleDirected(cycleGraph));

// MST tests
console.log("\n=== Minimum Spanning Tree ===");
const mstGraph = new Graph(4);
mstGraph.addEdge(0, 1, 10);
mstGraph.addEdge(0, 2, 6);
mstGraph.addEdge(0, 3, 5);
mstGraph.addEdge(1, 3, 15);
mstGraph.addEdge(2, 3, 4);

console.log("Kruskal MST:", kruskalMST(mstGraph));
console.log("Prim MST:", primMST(mstGraph));
```

## Key Insights

1. **Graph Representation**: Adjacency list is space-efficient for sparse graphs
2. **DFS vs BFS**: DFS uses less memory, BFS finds shortest path in unweighted graphs
3. **Shortest Path**: Dijkstra for non-negative weights, Bellman-Ford for negative weights
4. **Topological Sort**: Only possible for DAGs (Directed Acyclic Graphs)
5. **Cycle Detection**: Different approaches for directed vs undirected graphs
6. **MST**: Kruskal uses Union-Find, Prim uses priority queue

## Common Mistakes

1. **Not handling disconnected graphs** properly
2. **Incorrect cycle detection** logic
3. **Memory issues** with large graphs
4. **Incorrect priority queue** implementation
5. **Not updating distances** correctly in shortest path algorithms

## Related Problems

- [Number of Islands](../../../algorithms/Graphs/NumberOfIslands.md)
- [Course Schedule](../../../algorithms/Graphs/CourseSchedule.md)
- [Network Delay Time](../../../algorithms/Graphs/NetworkDelayTime.md)
- [Redundant Connection](../../../algorithms/Graphs/RedundantConnection.md)

## Interview Tips

1. **Choose appropriate representation** based on problem requirements
2. **Handle edge cases** (empty graph, single vertex, disconnected components)
3. **Explain time/space complexity** for each algorithm
4. **Discuss trade-offs** between different approaches
5. **Practice implementing** from scratch without reference
6. **Understand when to use** each algorithm
