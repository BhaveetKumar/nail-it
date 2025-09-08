# 🗄️ Vector Databases: Storing and Retrieving High-Dimensional Data

> **Complete guide to vector databases for AI applications, similarity search, and embeddings**

## 🎯 **Learning Objectives**

- Master vector database concepts and use cases
- Understand similarity search algorithms
- Implement vector storage and retrieval systems
- Apply vector databases to real-world AI problems
- Optimize vector database performance

## 📚 **Table of Contents**

1. [Vector Database Fundamentals](#vector-database-fundamentals)
2. [Similarity Search Algorithms](#similarity-search-algorithms)
3. [Vector Database Implementations](#vector-database-implementations)
4. [Real-world Applications](#real-world-applications)
5. [Performance Optimization](#performance-optimization)
6. [Interview Questions](#interview-questions)

---

## 🗄️ **Vector Database Fundamentals**

### **Concept**

Vector databases are specialized databases designed to store, index, and query high-dimensional vectors efficiently. They're essential for AI applications that work with embeddings, such as semantic search, recommendation systems, and similarity matching.

### **Key Features**

1. **High-Dimensional Storage**: Efficiently store vectors with hundreds or thousands of dimensions
2. **Similarity Search**: Fast nearest neighbor search using various distance metrics
3. **Scalability**: Handle millions or billions of vectors
4. **Real-time Updates**: Support dynamic insertion and deletion of vectors
5. **Metadata Support**: Store additional information alongside vectors

### **Use Cases**

- **Semantic Search**: Find similar documents, images, or products
- **Recommendation Systems**: Find similar users or items
- **Image Recognition**: Find similar images based on visual features
- **Natural Language Processing**: Find similar text based on embeddings
- **Anomaly Detection**: Identify outliers in high-dimensional data

---

## 🔍 **Similarity Search Algorithms**

### **1. Euclidean Distance**

**Formula**: `d(x, y) = √(Σ(x_i - y_i)²)`

**Code Example**:

```python
import numpy as np
from typing import List, Tuple
import time

class EuclideanDistanceSearch:
    def __init__(self, vectors: List[List[float]]):
        self.vectors = np.array(vectors)
        self.dimension = len(vectors[0]) if vectors else 0

    def search(self, query_vector: List[float], k: int = 10) -> List[Tuple[int, float]]:
        """Find k nearest neighbors using Euclidean distance"""
        query = np.array(query_vector)

        # Calculate distances
        distances = np.sqrt(np.sum((self.vectors - query) ** 2, axis=1))

        # Get k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]

        return [(idx, distances[idx]) for idx in nearest_indices]

    def add_vector(self, vector: List[float]) -> int:
        """Add a new vector to the database"""
        if self.dimension == 0:
            self.dimension = len(vector)
            self.vectors = np.array([vector])
        else:
            if len(vector) != self.dimension:
                raise ValueError("Vector dimension mismatch")
            self.vectors = np.vstack([self.vectors, vector])

        return len(self.vectors) - 1

# Example usage
vectors = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [1.1, 2.1, 3.1]
]

search_engine = EuclideanDistanceSearch(vectors)
query = [1.0, 2.0, 3.0]
results = search_engine.search(query, k=3)

print("Euclidean Distance Search Results:")
for idx, distance in results:
    print(f"Index: {idx}, Distance: {distance:.3f}, Vector: {vectors[idx]}")
```

### **2. Cosine Similarity**

**Formula**: `cos(x, y) = (x · y) / (||x|| × ||y||)`

**Code Example**:

```python
class CosineSimilaritySearch:
    def __init__(self, vectors: List[List[float]]):
        self.vectors = np.array(vectors)
        self.dimension = len(vectors[0]) if vectors else 0

        # Precompute norms for efficiency
        self.norms = np.linalg.norm(self.vectors, axis=1)

    def search(self, query_vector: List[float], k: int = 10) -> List[Tuple[int, float]]:
        """Find k nearest neighbors using cosine similarity"""
        query = np.array(query_vector)
        query_norm = np.linalg.norm(query)

        if query_norm == 0:
            raise ValueError("Query vector cannot be zero vector")

        # Calculate cosine similarities
        dot_products = np.dot(self.vectors, query)
        similarities = dot_products / (self.norms * query_norm)

        # Get k most similar vectors
        most_similar_indices = np.argsort(similarities)[::-1][:k]

        return [(idx, similarities[idx]) for idx in most_similar_indices]

    def add_vector(self, vector: List[float]) -> int:
        """Add a new vector to the database"""
        if self.dimension == 0:
            self.dimension = len(vector)
            self.vectors = np.array([vector])
            self.norms = np.array([np.linalg.norm(vector)])
        else:
            if len(vector) != self.dimension:
                raise ValueError("Vector dimension mismatch")
            self.vectors = np.vstack([self.vectors, vector])
            self.norms = np.append(self.norms, np.linalg.norm(vector))

        return len(self.vectors) - 1

# Example usage
cosine_search = CosineSimilaritySearch(vectors)
query = [1.0, 2.0, 3.0]
results = cosine_search.search(query, k=3)

print("\nCosine Similarity Search Results:")
for idx, similarity in results:
    print(f"Index: {idx}, Similarity: {similarity:.3f}, Vector: {vectors[idx]}")
```

### **3. Manhattan Distance**

**Formula**: `d(x, y) = Σ|x_i - y_i|`

**Code Example**:

```python
class ManhattanDistanceSearch:
    def __init__(self, vectors: List[List[float]]):
        self.vectors = np.array(vectors)
        self.dimension = len(vectors[0]) if vectors else 0

    def search(self, query_vector: List[float], k: int = 10) -> List[Tuple[int, float]]:
        """Find k nearest neighbors using Manhattan distance"""
        query = np.array(query_vector)

        # Calculate Manhattan distances
        distances = np.sum(np.abs(self.vectors - query), axis=1)

        # Get k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]

        return [(idx, distances[idx]) for idx in nearest_indices]

    def add_vector(self, vector: List[float]) -> int:
        """Add a new vector to the database"""
        if self.dimension == 0:
            self.dimension = len(vector)
            self.vectors = np.array([vector])
        else:
            if len(vector) != self.dimension:
                raise ValueError("Vector dimension mismatch")
            self.vectors = np.vstack([self.vectors, vector])

        return len(self.vectors) - 1

# Example usage
manhattan_search = ManhattanDistanceSearch(vectors)
query = [1.0, 2.0, 3.0]
results = manhattan_search.search(query, k=3)

print("\nManhattan Distance Search Results:")
for idx, distance in results:
    print(f"Index: {idx}, Distance: {distance:.3f}, Vector: {vectors[idx]}")
```

---

## 🏗️ **Vector Database Implementations**

### **1. Simple In-Memory Vector Database**

**Code Example**:

```python
import json
import pickle
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

class VectorDatabase:
    def __init__(self, dimension: int, distance_metric: str = "euclidean"):
        self.dimension = dimension
        self.distance_metric = distance_metric
        self.vectors = {}
        self.metadata = {}

        # Initialize search engine based on distance metric
        if distance_metric == "euclidean":
            self.search_engine = EuclideanDistanceSearch([])
        elif distance_metric == "cosine":
            self.search_engine = CosineSimilaritySearch([])
        elif distance_metric == "manhattan":
            self.search_engine = ManhattanDistanceSearch([])
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

    def add_vector(self, vector: List[float], metadata: Dict[str, Any] = None) -> str:
        """Add a vector with optional metadata"""
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension must be {self.dimension}")

        # Generate unique ID
        vector_id = str(uuid.uuid4())

        # Add to search engine
        self.search_engine.add_vector(vector)

        # Store vector and metadata
        self.vectors[vector_id] = vector
        self.metadata[vector_id] = {
            "created_at": datetime.now().isoformat(),
            "dimension": self.dimension,
            "distance_metric": self.distance_metric,
            **(metadata or {})
        }

        return vector_id

    def search(self, query_vector: List[float], k: int = 10,
               filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension must be {self.dimension}")

        # Get search results
        results = self.search_engine.search(query_vector, k)

        # Format results with metadata
        formatted_results = []
        for idx, distance in results:
            vector_id = list(self.vectors.keys())[idx]
            vector = self.vectors[vector_id]
            metadata = self.metadata[vector_id]

            # Apply metadata filter if provided
            if filter_metadata:
                if not all(metadata.get(key) == value for key, value in filter_metadata.items()):
                    continue

            formatted_results.append({
                "id": vector_id,
                "vector": vector,
                "distance": distance,
                "metadata": metadata
            })

        return formatted_results

    def get_vector(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Get vector by ID"""
        if vector_id not in self.vectors:
            return None

        return {
            "id": vector_id,
            "vector": self.vectors[vector_id],
            "metadata": self.metadata[vector_id]
        }

    def update_metadata(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a vector"""
        if vector_id not in self.vectors:
            return False

        self.metadata[vector_id].update(metadata)
        return True

    def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector"""
        if vector_id not in self.vectors:
            return False

        del self.vectors[vector_id]
        del self.metadata[vector_id]

        # Rebuild search engine (inefficient for large datasets)
        self._rebuild_search_engine()
        return True

    def _rebuild_search_engine(self):
        """Rebuild search engine after deletion"""
        vectors_list = list(self.vectors.values())

        if self.distance_metric == "euclidean":
            self.search_engine = EuclideanDistanceSearch(vectors_list)
        elif self.distance_metric == "cosine":
            self.search_engine = CosineSimilaritySearch(vectors_list)
        elif self.distance_metric == "manhattan":
            self.search_engine = ManhattanDistanceSearch(vectors_list)

    def save(self, filepath: str):
        """Save database to file"""
        data = {
            "dimension": self.dimension,
            "distance_metric": self.distance_metric,
            "vectors": self.vectors,
            "metadata": self.metadata
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        """Load database from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.dimension = data["dimension"]
        self.distance_metric = data["distance_metric"]
        self.vectors = data["vectors"]
        self.metadata = data["metadata"]

        # Rebuild search engine
        vectors_list = list(self.vectors.values())
        if self.distance_metric == "euclidean":
            self.search_engine = EuclideanDistanceSearch(vectors_list)
        elif self.distance_metric == "cosine":
            self.search_engine = CosineSimilaritySearch(vectors_list)
        elif self.distance_metric == "manhattan":
            self.search_engine = ManhattanDistanceSearch(vectors_list)

# Example usage
db = VectorDatabase(dimension=3, distance_metric="cosine")

# Add vectors with metadata
db.add_vector([1.0, 2.0, 3.0], {"category": "A", "label": "vector1"})
db.add_vector([4.0, 5.0, 6.0], {"category": "B", "label": "vector2"})
db.add_vector([7.0, 8.0, 9.0], {"category": "A", "label": "vector3"})
db.add_vector([1.1, 2.1, 3.1], {"category": "A", "label": "vector4"})

# Search for similar vectors
query = [1.0, 2.0, 3.0]
results = db.search(query, k=3)

print("Vector Database Search Results:")
for result in results:
    print(f"ID: {result['id']}, Distance: {result['distance']:.3f}, "
          f"Metadata: {result['metadata']}")

# Search with metadata filter
filtered_results = db.search(query, k=3, filter_metadata={"category": "A"})
print(f"\nFiltered Results (category A): {len(filtered_results)}")
```

### **2. FastAPI Vector Database Service**

**Code Example**:

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

app = FastAPI(title="Vector Database API", version="1.0.0")

# Global vector database
vector_db = VectorDatabase(dimension=384, distance_metric="cosine")  # Common embedding dimension

# Request/Response Models
class VectorRequest(BaseModel):
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    query_vector: List[float]
    k: int = 10
    filter_metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_results: int
    query_time: float

class VectorResponse(BaseModel):
    id: str
    vector: List[float]
    metadata: Dict[str, Any]

@app.post("/vectors", response_model=VectorResponse)
async def add_vector(request: VectorRequest):
    """Add a vector to the database"""
    try:
        vector_id = vector_db.add_vector(request.vector, request.metadata)
        return VectorResponse(
            id=vector_id,
            vector=request.vector,
            metadata=vector_db.metadata[vector_id]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search_vectors(request: SearchRequest):
    """Search for similar vectors"""
    try:
        import time
        start_time = time.time()

        results = vector_db.search(
            request.query_vector,
            k=request.k,
            filter_metadata=request.filter_metadata
        )

        query_time = time.time() - start_time

        return SearchResponse(
            results=results,
            total_results=len(results),
            query_time=query_time
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/vectors/{vector_id}", response_model=VectorResponse)
async def get_vector(vector_id: str):
    """Get a vector by ID"""
    result = vector_db.get_vector(vector_id)
    if not result:
        raise HTTPException(status_code=404, detail="Vector not found")

    return VectorResponse(**result)

@app.put("/vectors/{vector_id}/metadata")
async def update_metadata(vector_id: str, metadata: Dict[str, Any]):
    """Update metadata for a vector"""
    success = vector_db.update_metadata(vector_id, metadata)
    if not success:
        raise HTTPException(status_code=404, detail="Vector not found")

    return {"message": "Metadata updated successfully"}

@app.delete("/vectors/{vector_id}")
async def delete_vector(vector_id: str):
    """Delete a vector"""
    success = vector_db.delete_vector(vector_id)
    if not success:
        raise HTTPException(status_code=404, detail="Vector not found")

    return {"message": "Vector deleted successfully"}

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    return {
        "total_vectors": len(vector_db.vectors),
        "dimension": vector_db.dimension,
        "distance_metric": vector_db.distance_metric
    }

@app.post("/backup")
async def backup_database(background_tasks: BackgroundTasks):
    """Backup the database"""
    def backup():
        vector_db.save("backup.pkl")

    background_tasks.add_task(backup)
    return {"message": "Backup started"}

@app.post("/restore")
async def restore_database(background_tasks: BackgroundTasks):
    """Restore the database from backup"""
    def restore():
        vector_db.load("backup.pkl")

    background_tasks.add_task(restore)
    return {"message": "Restore started"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 🎯 **Real-world Applications**

### **1. Semantic Search System**

**Code Example**:

```python
import openai
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSearchSystem:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.vector_db = VectorDatabase(
            dimension=self.model.get_sentence_embedding_dimension(),
            distance_metric="cosine"
        )
        self.documents = {}

    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """Add a document to the search system"""
        # Generate embedding
        embedding = self.model.encode(text)

        # Add to vector database
        vector_id = self.vector_db.add_vector(embedding.tolist(), metadata)

        # Store document text
        self.documents[vector_id] = text

        return vector_id

    def search(self, query: str, k: int = 10,
               filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = self.model.encode(query)

        # Search vector database
        results = self.vector_db.search(
            query_embedding.tolist(),
            k=k,
            filter_metadata=filter_metadata
        )

        # Add document text to results
        for result in results:
            result["text"] = self.documents[result["id"]]

        return results

    def batch_add_documents(self, documents: List[Dict[str, Any]]):
        """Add multiple documents efficiently"""
        texts = [doc["text"] for doc in documents]
        embeddings = self.model.encode(texts)

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            vector_id = self.vector_db.add_vector(embedding.tolist(), doc.get("metadata"))
            self.documents[vector_id] = doc["text"]

# Example usage
search_system = SemanticSearchSystem()

# Add documents
documents = [
    {"text": "Machine learning is a subset of artificial intelligence.", "metadata": {"category": "AI"}},
    {"text": "Deep learning uses neural networks with multiple layers.", "metadata": {"category": "AI"}},
    {"text": "Python is a popular programming language for data science.", "metadata": {"category": "Programming"}},
    {"text": "Vector databases are optimized for similarity search.", "metadata": {"category": "Database"}},
    {"text": "Natural language processing deals with text and speech.", "metadata": {"category": "AI"}}
]

search_system.batch_add_documents(documents)

# Search for similar documents
query = "What is artificial intelligence?"
results = search_system.search(query, k=3)

print("Semantic Search Results:")
for result in results:
    print(f"Text: {result['text']}")
    print(f"Similarity: {result['distance']:.3f}")
    print(f"Metadata: {result['metadata']}")
    print("-" * 50)
```

### **2. Recommendation System**

**Code Example**:

```python
class RecommendationSystem:
    def __init__(self, user_dimension: int = 100, item_dimension: int = 100):
        self.user_db = VectorDatabase(dimension=user_dimension, distance_metric="cosine")
        self.item_db = VectorDatabase(dimension=item_dimension, distance_metric="cosine")
        self.user_items = {}  # Track user-item interactions

    def add_user(self, user_id: str, user_features: List[float], metadata: Dict[str, Any] = None):
        """Add a user to the system"""
        vector_id = self.user_db.add_vector(user_features, metadata)
        self.user_items[user_id] = []
        return vector_id

    def add_item(self, item_id: str, item_features: List[float], metadata: Dict[str, Any] = None):
        """Add an item to the system"""
        vector_id = self.item_db.add_vector(item_features, metadata)
        return vector_id

    def record_interaction(self, user_id: str, item_id: str, rating: float = 1.0):
        """Record user-item interaction"""
        if user_id not in self.user_items:
            self.user_items[user_id] = []

        self.user_items[user_id].append({"item_id": item_id, "rating": rating})

    def recommend_items(self, user_id: str, k: int = 10) -> List[Dict[str, Any]]:
        """Recommend items for a user"""
        if user_id not in self.user_items:
            return []

        # Get user vector
        user_vector = self.user_db.get_vector(user_id)
        if not user_vector:
            return []

        # Get items user hasn't interacted with
        interacted_items = {interaction["item_id"] for interaction in self.user_items[user_id]}

        # Search for similar items
        results = self.item_db.search(user_vector["vector"], k=k*2)  # Get more to filter

        # Filter out items user has already interacted with
        recommendations = []
        for result in results:
            if result["id"] not in interacted_items:
                recommendations.append(result)
                if len(recommendations) >= k:
                    break

        return recommendations

    def find_similar_users(self, user_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """Find similar users"""
        user_vector = self.user_db.get_vector(user_id)
        if not user_vector:
            return []

        results = self.user_db.search(user_vector["vector"], k=k+1)  # +1 to exclude self

        # Filter out the user themselves
        similar_users = [result for result in results if result["id"] != user_id]

        return similar_users[:k]

# Example usage
rec_system = RecommendationSystem()

# Add users
rec_system.add_user("user1", [1.0, 2.0, 3.0, 4.0, 5.0], {"age": 25, "gender": "M"})
rec_system.add_user("user2", [2.0, 3.0, 4.0, 5.0, 6.0], {"age": 30, "gender": "F"})
rec_system.add_user("user3", [1.5, 2.5, 3.5, 4.5, 5.5], {"age": 28, "gender": "M"})

# Add items
rec_system.add_item("item1", [1.0, 2.0, 3.0, 4.0, 5.0], {"category": "Action", "price": 29.99})
rec_system.add_item("item2", [2.0, 3.0, 4.0, 5.0, 6.0], {"category": "Comedy", "price": 19.99})
rec_system.add_item("item3", [1.5, 2.5, 3.5, 4.5, 5.5], {"category": "Drama", "price": 24.99})

# Record interactions
rec_system.record_interaction("user1", "item1", 5.0)
rec_system.record_interaction("user1", "item2", 4.0)
rec_system.record_interaction("user2", "item2", 5.0)
rec_system.record_interaction("user2", "item3", 4.5)

# Get recommendations
recommendations = rec_system.recommend_items("user1", k=2)
print("Recommendations for user1:")
for rec in recommendations:
    print(f"Item: {rec['id']}, Similarity: {rec['distance']:.3f}, Metadata: {rec['metadata']}")

# Find similar users
similar_users = rec_system.find_similar_users("user1", k=2)
print(f"\nSimilar users to user1:")
for user in similar_users:
    print(f"User: {user['id']}, Similarity: {user['distance']:.3f}, Metadata: {user['metadata']}")
```

---

## ⚡ **Performance Optimization**

### **1. Approximate Nearest Neighbor (ANN) Search**

**Code Example**:

```python
import random
from typing import List, Tuple
import heapq

class ANNSearch:
    def __init__(self, vectors: List[List[float]], num_trees: int = 10):
        self.vectors = np.array(vectors)
        self.num_trees = num_trees
        self.trees = []
        self._build_trees()

    def _build_trees(self):
        """Build multiple random projection trees"""
        for _ in range(self.num_trees):
            tree = self._build_tree(self.vectors)
            self.trees.append(tree)

    def _build_tree(self, vectors: np.ndarray, depth: int = 0, max_depth: int = 10):
        """Build a single random projection tree"""
        if len(vectors) <= 1 or depth >= max_depth:
            return {"vectors": vectors, "is_leaf": True}

        # Random projection
        projection = np.random.randn(vectors.shape[1])
        projections = np.dot(vectors, projection)

        # Split at median
        median = np.median(projections)
        left_mask = projections <= median
        right_mask = projections > median

        return {
            "projection": projection,
            "threshold": median,
            "left": self._build_tree(vectors[left_mask], depth + 1, max_depth),
            "right": self._build_tree(vectors[right_mask], depth + 1, max_depth),
            "is_leaf": False
        }

    def search(self, query_vector: List[float], k: int = 10) -> List[Tuple[int, float]]:
        """Search using multiple trees"""
        query = np.array(query_vector)
        candidates = set()

        # Search each tree
        for tree in self.trees:
            tree_candidates = self._search_tree(tree, query)
            candidates.update(tree_candidates)

        # Calculate exact distances for candidates
        distances = []
        for idx in candidates:
            distance = np.linalg.norm(self.vectors[idx] - query)
            distances.append((idx, distance))

        # Return k nearest
        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def _search_tree(self, tree: dict, query: np.ndarray) -> List[int]:
        """Search a single tree"""
        if tree["is_leaf"]:
            return list(range(len(tree["vectors"])))

        # Traverse tree
        projection_value = np.dot(query, tree["projection"])

        if projection_value <= tree["threshold"]:
            return self._search_tree(tree["left"], query)
        else:
            return self._search_tree(tree["right"], query)

# Example usage
# Generate random vectors
np.random.seed(42)
vectors = np.random.randn(1000, 50).tolist()

# Build ANN search
ann_search = ANNSearch(vectors, num_trees=10)

# Search
query = np.random.randn(50).tolist()
results = ann_search.search(query, k=5)

print("ANN Search Results:")
for idx, distance in results:
    print(f"Index: {idx}, Distance: {distance:.3f}")
```

### **2. Golang Implementation**

**Code Example**:

```go
package main

import (
    "fmt"
    "math"
    "sort"
)

type VectorDatabase struct {
    Vectors   [][]float64
    Metadata  []map[string]interface{}
    Dimension int
}

func NewVectorDatabase(dimension int) *VectorDatabase {
    return &VectorDatabase{
        Vectors:   make([][]float64, 0),
        Metadata:  make([]map[string]interface{}, 0),
        Dimension: dimension,
    }
}

func (db *VectorDatabase) AddVector(vector []float64, metadata map[string]interface{}) int {
    if len(vector) != db.Dimension {
        panic("Vector dimension mismatch")
    }

    db.Vectors = append(db.Vectors, vector)
    db.Metadata = append(db.Metadata, metadata)

    return len(db.Vectors) - 1
}

func (db *VectorDatabase) EuclideanDistance(v1, v2 []float64) float64 {
    sum := 0.0
    for i := range v1 {
        diff := v1[i] - v2[i]
        sum += diff * diff
    }
    return math.Sqrt(sum)
}

func (db *VectorDatabase) CosineSimilarity(v1, v2 []float64) float64 {
    dotProduct := 0.0
    norm1 := 0.0
    norm2 := 0.0

    for i := range v1 {
        dotProduct += v1[i] * v2[i]
        norm1 += v1[i] * v1[i]
        norm2 += v2[i] * v2[i]
    }

    if norm1 == 0 || norm2 == 0 {
        return 0
    }

    return dotProduct / (math.Sqrt(norm1) * math.Sqrt(norm2))
}

func (db *VectorDatabase) Search(query []float64, k int, useCosine bool) []SearchResult {
    results := make([]SearchResult, 0, len(db.Vectors))

    for i, vector := range db.Vectors {
        var distance float64
        if useCosine {
            distance = 1 - db.CosineSimilarity(query, vector) // Convert to distance
        } else {
            distance = db.EuclideanDistance(query, vector)
        }

        results = append(results, SearchResult{
            Index:    i,
            Distance: distance,
            Vector:   vector,
            Metadata: db.Metadata[i],
        })
    }

    // Sort by distance
    sort.Slice(results, func(i, j int) bool {
        return results[i].Distance < results[j].Distance
    })

    // Return top k
    if k > len(results) {
        k = len(results)
    }

    return results[:k]
}

type SearchResult struct {
    Index    int
    Distance float64
    Vector   []float64
    Metadata map[string]interface{}
}

func main() {
    // Create vector database
    db := NewVectorDatabase(3)

    // Add vectors
    db.AddVector([]float64{1.0, 2.0, 3.0}, map[string]interface{}{"label": "vector1"})
    db.AddVector([]float64{4.0, 5.0, 6.0}, map[string]interface{}{"label": "vector2"})
    db.AddVector([]float64{7.0, 8.0, 9.0}, map[string]interface{}{"label": "vector3"})

    // Search
    query := []float64{1.0, 2.0, 3.0}
    results := db.Search(query, 2, false) // Use Euclidean distance

    fmt.Println("Search Results:")
    for _, result := range results {
        fmt.Printf("Index: %d, Distance: %.3f, Label: %v\n",
            result.Index, result.Distance, result.Metadata["label"])
    }
}
```

---

## 🎯 **Interview Questions**

### **1. What are vector databases and when would you use them?**

**Answer:**
Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently:

- **Use Cases**: Semantic search, recommendation systems, image similarity, NLP applications
- **Key Features**: Fast similarity search, high-dimensional storage, real-time updates
- **When to Use**: When you need to find similar items based on embeddings or features
- **Benefits**: Better performance than traditional databases for similarity search

### **2. What are the different distance metrics used in vector databases?**

**Answer:**

- **Euclidean Distance**: `√(Σ(x_i - y_i)²)` - Good for continuous features
- **Cosine Similarity**: `(x·y)/(||x||×||y||)` - Good for text embeddings, ignores magnitude
- **Manhattan Distance**: `Σ|x_i - y_i|` - Good for categorical features
- **Dot Product**: `x·y` - Simple but sensitive to vector magnitude
- **Jaccard Similarity**: For binary vectors, measures overlap

### **3. How do you optimize vector database performance?**

**Answer:**

- **Indexing**: Use specialized indexes like HNSW, IVF, or LSH
- **Approximate Search**: Trade accuracy for speed with ANN algorithms
- **Caching**: Cache frequently accessed vectors
- **Batch Operations**: Process multiple vectors together
- **Compression**: Use quantization to reduce memory usage
- **Parallel Processing**: Distribute search across multiple cores

### **4. What are the challenges of scaling vector databases?**

**Answer:**

- **Memory Usage**: High-dimensional vectors consume significant memory
- **Search Complexity**: Exact search is O(n) for each query
- **Index Maintenance**: Updating indexes for new vectors is expensive
- **Data Distribution**: Balancing load across multiple nodes
- **Consistency**: Maintaining consistency in distributed systems
- **Query Latency**: Ensuring low latency for real-time applications

### **5. How do you handle updates and deletions in vector databases?**

**Answer:**

- **Lazy Deletion**: Mark vectors as deleted, rebuild index periodically
- **Delta Updates**: Maintain separate delta index for recent changes
- **Versioning**: Keep multiple versions of vectors
- **Rebuilding**: Periodically rebuild the entire index
- **Incremental Updates**: Update only affected parts of the index
- **Batch Updates**: Accumulate changes and apply in batches

---

**🎉 Vector databases are essential for modern AI applications that require efficient similarity search!**
