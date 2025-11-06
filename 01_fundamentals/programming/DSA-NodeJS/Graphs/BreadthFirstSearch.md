---
# Auto-generated front matter
Title: Breadthfirstsearch
LastUpdated: 2025-11-06T20:45:58.794313
Tags: []
Status: draft
---

# 🔍 Breadth-First Search (BFS) - Graph Problem

> **Master BFS algorithm for graph traversal and shortest path problems**

## 📚 Overview

Breadth-First Search (BFS) is a graph traversal algorithm that explores nodes level by level. It's particularly useful for finding shortest paths in unweighted graphs and level-order traversal.

## 🎯 Graph Representation

```javascript
// Adjacency List Representation
class Graph {
  constructor() {
    this.adjacencyList = new Map();
  }

  addVertex(vertex) {
    if (!this.adjacencyList.has(vertex)) {
      this.adjacencyList.set(vertex, []);
    }
  }

  addEdge(vertex1, vertex2) {
    this.addVertex(vertex1);
    this.addVertex(vertex2);

    this.adjacencyList.get(vertex1).push(vertex2);
    this.adjacencyList.get(vertex2).push(vertex1); // For undirected graph
  }

  getNeighbors(vertex) {
    return this.adjacencyList.get(vertex) || [];
  }

  getAllVertices() {
    return Array.from(this.adjacencyList.keys());
  }
}

// Create sample graph
function createSampleGraph() {
  const graph = new Graph();

  // Add edges
  graph.addEdge("A", "B");
  graph.addEdge("A", "C");
  graph.addEdge("B", "D");
  graph.addEdge("B", "E");
  graph.addEdge("C", "F");
  graph.addEdge("D", "G");
  graph.addEdge("E", "H");
  graph.addEdge("F", "H");

  return graph;
}

/*
Graph Structure:
    A
   / \
  B   C
 /|   |
D E   F
|     |
G     H
 \   /
   H
*/
```

## 🔍 Basic BFS Implementation

### **1. Simple BFS Traversal**

```javascript
/**
 * Basic BFS Traversal
 * Time Complexity: O(V + E)
 * Space Complexity: O(V)
 */
function bfs(graph, startVertex) {
  const visited = new Set();
  const queue = [startVertex];
  const result = [];

  visited.add(startVertex);

  while (queue.length > 0) {
    const currentVertex = queue.shift();
    result.push(currentVertex);

    const neighbors = graph.getNeighbors(currentVertex);

    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push(neighbor);
      }
    }
  }

  return result;
}

// Test
const graph = createSampleGraph();
console.log("BFS Traversal from A:", bfs(graph, "A")); // ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
```

### **2. BFS with Level Information**

```javascript
/**
 * BFS with Level Information
 * Returns nodes grouped by their distance from start
 */
function bfsWithLevels(graph, startVertex) {
  const visited = new Set();
  const queue = [{ vertex: startVertex, level: 0 }];
  const result = [];

  visited.add(startVertex);

  while (queue.length > 0) {
    const { vertex, level } = queue.shift();

    // Ensure we have enough levels in result array
    if (!result[level]) {
      result[level] = [];
    }
    result[level].push(vertex);

    const neighbors = graph.getNeighbors(vertex);

    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push({ vertex: neighbor, level: level + 1 });
      }
    }
  }

  return result;
}

// Test
console.log("BFS with Levels:", bfsWithLevels(graph, "A"));
// [['A'], ['B', 'C'], ['D', 'E', 'F'], ['G', 'H']]
```

### **3. BFS with Path Tracking**

```javascript
/**
 * BFS with Path Tracking
 * Keeps track of the path to each vertex
 */
function bfsWithPath(graph, startVertex, targetVertex) {
  const visited = new Set();
  const queue = [{ vertex: startVertex, path: [startVertex] }];

  visited.add(startVertex);

  while (queue.length > 0) {
    const { vertex, path } = queue.shift();

    if (vertex === targetVertex) {
      return path;
    }

    const neighbors = graph.getNeighbors(vertex);

    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push({
          vertex: neighbor,
          path: [...path, neighbor],
        });
      }
    }
  }

  return null; // No path found
}

// Test
console.log("Path from A to H:", bfsWithPath(graph, "A", "H")); // ['A', 'B', 'E', 'H']
```

## 🛤️ Shortest Path Problems

### **1. Shortest Path in Unweighted Graph**

```javascript
/**
 * Find shortest path between two vertices
 * Returns the shortest path and its length
 */
function shortestPath(graph, startVertex, targetVertex) {
  const visited = new Set();
  const queue = [{ vertex: startVertex, path: [startVertex], distance: 0 }];

  visited.add(startVertex);

  while (queue.length > 0) {
    const { vertex, path, distance } = queue.shift();

    if (vertex === targetVertex) {
      return { path, distance };
    }

    const neighbors = graph.getNeighbors(vertex);

    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push({
          vertex: neighbor,
          path: [...path, neighbor],
          distance: distance + 1,
        });
      }
    }
  }

  return null; // No path found
}

// Test
const pathResult = shortestPath(graph, "A", "H");
console.log("Shortest path from A to H:", pathResult);
// { path: ['A', 'B', 'E', 'H'], distance: 3 }
```

### **2. All Shortest Paths from Source**

```javascript
/**
 * Find shortest distances from source to all vertices
 * Returns a map of vertex -> distance
 */
function allShortestPaths(graph, startVertex) {
  const distances = new Map();
  const visited = new Set();
  const queue = [{ vertex: startVertex, distance: 0 }];

  visited.add(startVertex);
  distances.set(startVertex, 0);

  while (queue.length > 0) {
    const { vertex, distance } = queue.shift();

    const neighbors = graph.getNeighbors(vertex);

    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        distances.set(neighbor, distance + 1);
        queue.push({ vertex: neighbor, distance: distance + 1 });
      }
    }
  }

  return distances;
}

// Test
const allDistances = allShortestPaths(graph, "A");
console.log("All distances from A:", Object.fromEntries(allDistances));
// { A: 0, B: 1, C: 1, D: 2, E: 2, F: 2, G: 3, H: 3 }
```

## 🎯 Matrix BFS Problems

### **1. Number of Islands**

```javascript
/**
 * Number of Islands
 * Count the number of connected components of 1s in a 2D matrix
 */
function numIslands(grid) {
  if (!grid || grid.length === 0) return 0;

  const rows = grid.length;
  const cols = grid[0].length;
  let islands = 0;

  function bfs(startRow, startCol) {
    const queue = [[startRow, startCol]];
    grid[startRow][startCol] = "0"; // Mark as visited

    const directions = [
      [-1, 0],
      [1, 0],
      [0, -1],
      [0, 1],
    ]; // up, down, left, right

    while (queue.length > 0) {
      const [row, col] = queue.shift();

      for (const [dr, dc] of directions) {
        const newRow = row + dr;
        const newCol = col + dc;

        if (
          newRow >= 0 &&
          newRow < rows &&
          newCol >= 0 &&
          newCol < cols &&
          grid[newRow][newCol] === "1"
        ) {
          grid[newRow][newCol] = "0";
          queue.push([newRow, newCol]);
        }
      }
    }
  }

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      if (grid[i][j] === "1") {
        islands++;
        bfs(i, j);
      }
    }
  }

  return islands;
}

// Test
const grid = [
  ["1", "1", "1", "1", "0"],
  ["1", "1", "0", "1", "0"],
  ["1", "1", "0", "0", "0"],
  ["0", "0", "0", "0", "0"],
];
console.log("Number of islands:", numIslands(grid)); // 1
```

### **2. Rotting Oranges**

```javascript
/**
 * Rotting Oranges
 * Find minimum time for all fresh oranges to rot
 */
function orangesRotting(grid) {
  if (!grid || grid.length === 0) return -1;

  const rows = grid.length;
  const cols = grid[0].length;
  const queue = [];
  let freshOranges = 0;

  // Find all rotten oranges and count fresh ones
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      if (grid[i][j] === 2) {
        queue.push([i, j, 0]); // [row, col, time]
      } else if (grid[i][j] === 1) {
        freshOranges++;
      }
    }
  }

  if (freshOranges === 0) return 0;

  const directions = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
  ];
  let maxTime = 0;

  while (queue.length > 0) {
    const [row, col, time] = queue.shift();
    maxTime = Math.max(maxTime, time);

    for (const [dr, dc] of directions) {
      const newRow = row + dr;
      const newCol = col + dc;

      if (
        newRow >= 0 &&
        newRow < rows &&
        newCol >= 0 &&
        newCol < cols &&
        grid[newRow][newCol] === 1
      ) {
        grid[newRow][newCol] = 2;
        freshOranges--;
        queue.push([newRow, newCol, time + 1]);
      }
    }
  }

  return freshOranges === 0 ? maxTime : -1;
}

// Test
const orangeGrid = [
  [2, 1, 1],
  [1, 1, 0],
  [0, 1, 1],
];
console.log("Time to rot all oranges:", orangesRotting(orangeGrid)); // 4
```

## 🎯 Advanced BFS Applications

### **1. Word Ladder**

```javascript
/**
 * Word Ladder
 * Find shortest transformation sequence from beginWord to endWord
 */
function ladderLength(beginWord, endWord, wordList) {
  const wordSet = new Set(wordList);

  if (!wordSet.has(endWord)) return 0;

  const queue = [{ word: beginWord, length: 1 }];
  const visited = new Set([beginWord]);

  while (queue.length > 0) {
    const { word, length } = queue.shift();

    if (word === endWord) {
      return length;
    }

    // Try all possible one-character changes
    for (let i = 0; i < word.length; i++) {
      for (let c = 97; c <= 122; c++) {
        // 'a' to 'z'
        const newWord =
          word.slice(0, i) + String.fromCharCode(c) + word.slice(i + 1);

        if (wordSet.has(newWord) && !visited.has(newWord)) {
          visited.add(newWord);
          queue.push({ word: newWord, length: length + 1 });
        }
      }
    }
  }

  return 0;
}

// Test
const wordList = ["hot", "dot", "dog", "lot", "log", "cog"];
console.log("Word ladder length:", ladderLength("hit", "cog", wordList)); // 5
```

### **2. Binary Tree Level Order Traversal**

```javascript
/**
 * Binary Tree Level Order Traversal using BFS
 */
class TreeNode {
  constructor(val, left = null, right = null) {
    this.val = val;
    this.left = left;
    this.right = right;
  }
}

function levelOrder(root) {
  if (!root) return [];

  const result = [];
  const queue = [root];

  while (queue.length > 0) {
    const levelSize = queue.length;
    const currentLevel = [];

    for (let i = 0; i < levelSize; i++) {
      const node = queue.shift();
      currentLevel.push(node.val);

      if (node.left) queue.push(node.left);
      if (node.right) queue.push(node.right);
    }

    result.push(currentLevel);
  }

  return result;
}

// Test
const tree = new TreeNode(3);
tree.left = new TreeNode(9);
tree.right = new TreeNode(20);
tree.right.left = new TreeNode(15);
tree.right.right = new TreeNode(7);

console.log("Level order traversal:", levelOrder(tree)); // [[3], [9, 20], [15, 7]]
```

## 🧪 Comprehensive Test Suite

```javascript
// Test all BFS implementations
function runBFSTests() {
  console.log("=== BFS Algorithm Tests ===\n");

  const graph = createSampleGraph();

  console.log("Graph Structure:");
  console.log("    A");
  console.log("   / \\");
  console.log("  B   C");
  console.log(" /|   |");
  console.log("D E   F");
  console.log("|     |");
  console.log("G     H");
  console.log(" \\   /");
  console.log("   H\n");

  console.log("1. Basic BFS Traversal:");
  console.log("From A:", bfs(graph, "A"));
  console.log("From B:", bfs(graph, "B"));
  console.log();

  console.log("2. BFS with Levels:");
  console.log(bfsWithLevels(graph, "A"));
  console.log();

  console.log("3. Shortest Path:");
  console.log("A to H:", shortestPath(graph, "A", "H"));
  console.log("A to G:", shortestPath(graph, "A", "G"));
  console.log();

  console.log("4. All Shortest Paths from A:");
  console.log(Object.fromEntries(allShortestPaths(graph, "A")));
  console.log();

  console.log("5. Matrix BFS - Number of Islands:");
  const testGrid = [
    ["1", "1", "0", "0", "0"],
    ["1", "1", "0", "0", "0"],
    ["0", "0", "1", "0", "0"],
    ["0", "0", "0", "1", "1"],
  ];
  console.log("Grid:", testGrid);
  console.log("Number of islands:", numIslands([...testGrid]));
  console.log();

  console.log("6. Word Ladder:");
  const words = ["hot", "dot", "dog", "lot", "log", "cog"];
  console.log("Word list:", words);
  console.log("Ladder length (hit -> cog):", ladderLength("hit", "cog", words));
}

runBFSTests();
```

## 📊 Performance Analysis

```javascript
// Performance comparison
function performanceTest() {
  // Create a large graph
  function createLargeGraph(vertices) {
    const graph = new Graph();

    for (let i = 0; i < vertices; i++) {
      graph.addVertex(i);
    }

    // Add random edges
    for (let i = 0; i < vertices * 2; i++) {
      const v1 = Math.floor(Math.random() * vertices);
      const v2 = Math.floor(Math.random() * vertices);
      if (v1 !== v2) {
        graph.addEdge(v1, v2);
      }
    }

    return graph;
  }

  const largeGraph = createLargeGraph(1000);

  console.log("Performance Test with 1000 vertices:");

  console.time("BFS Traversal");
  bfs(largeGraph, 0);
  console.timeEnd("BFS Traversal");

  console.time("All Shortest Paths");
  allShortestPaths(largeGraph, 0);
  console.timeEnd("All Shortest Paths");
}

// Uncomment to run performance test
// performanceTest();
```

## 🎯 Interview Tips

### **Key Points to Remember:**

1. **BFS uses a queue** (FIFO) for level-by-level traversal
2. **Time Complexity**: O(V + E) for adjacency list, O(V²) for adjacency matrix
3. **Space Complexity**: O(V) for visited set and queue
4. **Optimal for shortest path** in unweighted graphs

### **When to Use BFS:**

- Shortest path in unweighted graphs
- Level-order traversal
- Finding connected components
- Minimum steps problems
- Level-by-level processing

### **Common Patterns:**

1. **Matrix BFS**: Use directions array for 4/8 directions
2. **Tree BFS**: Process level by level
3. **Graph BFS**: Track visited nodes to avoid cycles
4. **Path BFS**: Store path information in queue

### **Follow-up Questions:**

1. How would you handle weighted graphs?
2. What if the graph is very large?
3. How to find all shortest paths?
4. Memory optimization techniques?

---

**🎉 Master BFS to solve graph traversal and shortest path problems efficiently!**

**Good luck with your coding interviews! 🚀**
