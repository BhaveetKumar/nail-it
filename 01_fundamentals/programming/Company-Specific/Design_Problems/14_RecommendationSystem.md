---
# Auto-generated front matter
Title: 14 Recommendationsystem
LastUpdated: 2025-11-06T20:45:58.772690
Tags: []
Status: draft
---

# 14. Recommendation System - Content Discovery Engine

## Title & Summary
Design and implement a recommendation system using Node.js that provides personalized content recommendations with collaborative filtering, content-based filtering, and real-time recommendation updates.

## Problem Statement

Build a comprehensive recommendation system that:

1. **Personalized Recommendations**: Provide tailored content for each user
2. **Multiple Algorithms**: Support collaborative and content-based filtering
3. **Real-time Updates**: Update recommendations based on user interactions
4. **Scalability**: Handle millions of users and items
5. **Performance**: Sub-100ms recommendation generation
6. **A/B Testing**: Support multiple recommendation strategies

## Requirements & Constraints

### Functional Requirements
- User-item interaction tracking
- Collaborative filtering (user-based and item-based)
- Content-based filtering
- Real-time recommendation updates
- Recommendation caching and optimization
- User preference learning
- Cold start problem handling

### Non-Functional Requirements
- **Latency**: < 100ms for recommendation generation
- **Throughput**: 10,000+ recommendations per second
- **Availability**: 99.9% uptime
- **Scalability**: Support 10M+ users and 100M+ items
- **Accuracy**: High recommendation relevance
- **Real-time**: Update recommendations within 1 second

## API / Interfaces

### REST Endpoints

```javascript
// Recommendations
GET    /api/recommendations/{userId}
GET    /api/recommendations/{userId}/similar
POST   /api/recommendations/refresh

// User Interactions
POST   /api/interactions
GET    /api/interactions/{userId}
PUT    /api/interactions/{interactionId}

// Content Management
GET    /api/items/{itemId}
POST   /api/items
PUT    /api/items/{itemId}
GET    /api/items/similar/{itemId}

// Analytics
GET    /api/analytics/recommendations
GET    /api/analytics/performance
GET    /api/analytics/user-behavior
```

### Request/Response Examples

```json
// Get Recommendations
GET /api/recommendations/user_123

// Response
{
  "success": true,
  "data": {
    "userId": "user_123",
    "recommendations": [
      {
        "itemId": "item_456",
        "score": 0.95,
        "reason": "Users with similar preferences also liked this",
        "category": "electronics",
        "price": 299.99
      }
    ],
    "algorithm": "hybrid",
    "generatedAt": "2024-01-15T10:30:00Z",
    "cacheHit": true
  }
}

// User Interaction
POST /api/interactions
{
  "userId": "user_123",
  "itemId": "item_456",
  "interactionType": "view",
  "timestamp": "2024-01-15T10:30:00Z",
  "metadata": {
    "duration": 30,
    "source": "recommendation"
  }
}
```

## Data Model

### Core Entities

```javascript
// User Entity
class User {
  constructor(userId, profile) {
    this.id = userId;
    this.profile = profile;
    this.preferences = {};
    this.interactions = [];
    this.createdAt = new Date();
    this.lastActive = new Date();
  }
}

// Item Entity
class Item {
  constructor(itemId, metadata) {
    this.id = itemId;
    this.metadata = metadata;
    this.features = {};
    this.ratings = [];
    this.createdAt = new Date();
    this.updatedAt = new Date();
  }
}

// Interaction Entity
class Interaction {
  constructor(userId, itemId, type, metadata) {
    this.id = this.generateID();
    this.userId = userId;
    this.itemId = itemId;
    this.type = type; // 'view', 'like', 'purchase', 'rating'
    this.metadata = metadata;
    this.timestamp = new Date();
    this.score = this.calculateScore();
  }
}

// Recommendation Entity
class Recommendation {
  constructor(userId, itemId, score, algorithm) {
    this.id = this.generateID();
    this.userId = userId;
    this.itemId = itemId;
    this.score = score;
    this.algorithm = algorithm;
    this.generatedAt = new Date();
    this.expiresAt = new Date(Date.now() + 3600000); // 1 hour
  }
}
```

## Approach Overview

### Simple Solution (MVP)
1. Basic collaborative filtering
2. Simple user-item matrix
3. No real-time updates
4. In-memory storage

### Production-Ready Design
1. **Hybrid Approach**: Combine multiple recommendation algorithms
2. **Real-time Processing**: Stream processing for user interactions
3. **Scalable Architecture**: Distributed recommendation generation
4. **Caching Strategy**: Multi-level caching for performance
5. **A/B Testing**: Support multiple recommendation strategies
6. **Machine Learning**: Advanced ML models for recommendations

## Detailed Design

### Core Service Implementation

```javascript
const EventEmitter = require("events");
const { v4: uuidv4 } = require("uuid");

class RecommendationService extends EventEmitter {
  constructor() {
    super();
    this.users = new Map();
    this.items = new Map();
    this.interactions = new Map();
    this.recommendations = new Map();
    this.userItemMatrix = new Map();
    this.itemSimilarity = new Map();
    this.userSimilarity = new Map();
    
    // Algorithms
    this.algorithms = {
      collaborative: new CollaborativeFiltering(),
      contentBased: new ContentBasedFiltering(),
      hybrid: new HybridRecommendation()
    };
    
    // Start background tasks
    this.startRecommendationGeneration();
    this.startSimilarityCalculation();
    this.startCacheRefresh();
  }

  // User Interaction Tracking
  async recordInteraction(interactionData) {
    try {
      const { userId, itemId, type, metadata } = interactionData;
      
      const interaction = new Interaction(userId, itemId, type, metadata);
      this.interactions.set(interaction.id, interaction);
      
      // Update user-item matrix
      this.updateUserItemMatrix(userId, itemId, interaction.score);
      
      // Update user preferences
      this.updateUserPreferences(userId, itemId, type);
      
      // Invalidate user recommendations
      this.invalidateUserRecommendations(userId);
      
      this.emit("interactionRecorded", { interaction, userId, itemId });
      
      return interaction;
      
    } catch (error) {
      console.error("Interaction recording error:", error);
      throw error;
    }
  }

  // Recommendation Generation
  async generateRecommendations(userId, options = {}) {
    try {
      const { algorithm = "hybrid", limit = 10, excludeInteracted = true } = options;
      
      // Check cache first
      const cached = this.getCachedRecommendations(userId);
      if (cached && !this.isExpired(cached)) {
        return cached.recommendations.slice(0, limit);
      }
      
      // Generate new recommendations
      const recommendations = await this.algorithms[algorithm].generate(
        userId, 
        { limit, excludeInteracted }
      );
      
      // Cache recommendations
      this.cacheRecommendations(userId, recommendations);
      
      this.emit("recommendationsGenerated", { userId, recommendations, algorithm });
      
      return recommendations;
      
    } catch (error) {
      console.error("Recommendation generation error:", error);
      throw error;
    }
  }

  // Collaborative Filtering
  async generateCollaborativeRecommendations(userId, options) {
    try {
      const { limit = 10, excludeInteracted = true } = options;
      
      // Find similar users
      const similarUsers = this.findSimilarUsers(userId, 50);
      
      // Get items liked by similar users
      const candidateItems = new Map();
      
      for (const [similarUserId, similarity] of similarUsers) {
        const userItems = this.getUserItems(similarUserId);
        
        for (const [itemId, rating] of userItems) {
          if (excludeInteracted && this.hasUserInteracted(userId, itemId)) {
            continue;
          }
          
          const score = candidateItems.get(itemId) || 0;
          candidateItems.set(itemId, score + (rating * similarity));
        }
      }
      
      // Sort by score and return top items
      const recommendations = Array.from(candidateItems.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, limit)
        .map(([itemId, score]) => ({
          itemId,
          score: this.normalizeScore(score),
          reason: "Users with similar preferences also liked this",
          algorithm: "collaborative"
        }));
      
      return recommendations;
      
    } catch (error) {
      console.error("Collaborative filtering error:", error);
      throw error;
    }
  }

  // Content-Based Filtering
  async generateContentBasedRecommendations(userId, options) {
    try {
      const { limit = 10, excludeInteracted = true } = options;
      
      // Get user's liked items
      const userItems = this.getUserItems(userId);
      const userPreferences = this.extractUserPreferences(userItems);
      
      // Find similar items
      const candidateItems = new Map();
      
      for (const [itemId, rating] of userItems) {
        const similarItems = this.findSimilarItems(itemId, 20);
        
        for (const [similarItemId, similarity] of similarItems) {
          if (excludeInteracted && this.hasUserInteracted(userId, similarItemId)) {
            continue;
          }
          
          const score = candidateItems.get(similarItemId) || 0;
          candidateItems.set(similarItemId, score + (rating * similarity));
        }
      }
      
      // Sort by score and return top items
      const recommendations = Array.from(candidateItems.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, limit)
        .map(([itemId, score]) => ({
          itemId,
          score: this.normalizeScore(score),
          reason: "Similar to items you've liked",
          algorithm: "content-based"
        }));
      
      return recommendations;
      
    } catch (error) {
      console.error("Content-based filtering error:", error);
      throw error;
    }
  }

  // Hybrid Recommendation
  async generateHybridRecommendations(userId, options) {
    try {
      const { limit = 10 } = options;
      
      // Generate recommendations from both algorithms
      const [collaborative, contentBased] = await Promise.all([
        this.generateCollaborativeRecommendations(userId, options),
        this.generateContentBasedRecommendations(userId, options)
      ]);
      
      // Combine and rank recommendations
      const combined = new Map();
      
      // Add collaborative recommendations with weight 0.6
      for (const rec of collaborative) {
        const score = combined.get(rec.itemId) || 0;
        combined.set(rec.itemId, score + (rec.score * 0.6));
      }
      
      // Add content-based recommendations with weight 0.4
      for (const rec of contentBased) {
        const score = combined.get(rec.itemId) || 0;
        combined.set(rec.itemId, score + (rec.score * 0.4));
      }
      
      // Sort by combined score
      const recommendations = Array.from(combined.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, limit)
        .map(([itemId, score]) => ({
          itemId,
          score: this.normalizeScore(score),
          reason: "Combined collaborative and content-based filtering",
          algorithm: "hybrid"
        }));
      
      return recommendations;
      
    } catch (error) {
      console.error("Hybrid recommendation error:", error);
      throw error;
    }
  }

  // Similarity Calculation
  calculateUserSimilarity(user1, user2) {
    const items1 = this.getUserItems(user1);
    const items2 = this.getUserItems(user2);
    
    const commonItems = new Set();
    for (const itemId of items1.keys()) {
      if (items2.has(itemId)) {
        commonItems.add(itemId);
      }
    }
    
    if (commonItems.size === 0) return 0;
    
    let numerator = 0;
    let sum1 = 0, sum2 = 0;
    let sum1Sq = 0, sum2Sq = 0;
    
    for (const itemId of commonItems) {
      const rating1 = items1.get(itemId);
      const rating2 = items2.get(itemId);
      
      numerator += rating1 * rating2;
      sum1 += rating1;
      sum2 += rating2;
      sum1Sq += rating1 * rating1;
      sum2Sq += rating2 * rating2;
    }
    
    const denominator = Math.sqrt(sum1Sq - (sum1 * sum1 / commonItems.size)) *
                       Math.sqrt(sum2Sq - (sum2 * sum2 / commonItems.size));
    
    return denominator === 0 ? 0 : numerator / denominator;
  }

  calculateItemSimilarity(item1, item2) {
    const users1 = this.getItemUsers(item1);
    const users2 = this.getItemUsers(item2);
    
    const commonUsers = new Set();
    for (const userId of users1.keys()) {
      if (users2.has(userId)) {
        commonUsers.add(userId);
      }
    }
    
    if (commonUsers.size === 0) return 0;
    
    let numerator = 0;
    let sum1 = 0, sum2 = 0;
    let sum1Sq = 0, sum2Sq = 0;
    
    for (const userId of commonUsers) {
      const rating1 = users1.get(userId);
      const rating2 = users2.get(userId);
      
      numerator += rating1 * rating2;
      sum1 += rating1;
      sum2 += rating2;
      sum1Sq += rating1 * rating1;
      sum2Sq += rating2 * rating2;
    }
    
    const denominator = Math.sqrt(sum1Sq - (sum1 * sum1 / commonUsers.size)) *
                       Math.sqrt(sum2Sq - (sum2 * sum2 / commonUsers.size));
    
    return denominator === 0 ? 0 : numerator / denominator;
  }

  // Background Tasks
  startRecommendationGeneration() {
    setInterval(() => {
      this.generateRecommendationsForActiveUsers();
    }, 300000); // Run every 5 minutes
  }

  async generateRecommendationsForActiveUsers() {
    const activeUsers = Array.from(this.users.values())
      .filter(user => Date.now() - user.lastActive.getTime() < 86400000) // Last 24 hours
      .slice(0, 1000); // Limit to 1000 users per batch
    
    for (const user of activeUsers) {
      try {
        await this.generateRecommendations(user.id, { algorithm: "hybrid" });
      } catch (error) {
        console.error(`Failed to generate recommendations for user ${user.id}:`, error);
      }
    }
  }

  startSimilarityCalculation() {
    setInterval(() => {
      this.calculateSimilarities();
    }, 3600000); // Run every hour
  }

  async calculateSimilarities() {
    // Calculate user similarities
    const userIds = Array.from(this.users.keys());
    for (let i = 0; i < userIds.length; i++) {
      for (let j = i + 1; j < userIds.length; j++) {
        const similarity = this.calculateUserSimilarity(userIds[i], userIds[j]);
        if (similarity > 0.1) { // Only store significant similarities
          this.userSimilarity.set(`${userIds[i]}-${userIds[j]}`, similarity);
        }
      }
    }
    
    // Calculate item similarities
    const itemIds = Array.from(this.items.keys());
    for (let i = 0; i < itemIds.length; i++) {
      for (let j = i + 1; j < itemIds.length; j++) {
        const similarity = this.calculateItemSimilarity(itemIds[i], itemIds[j]);
        if (similarity > 0.1) { // Only store significant similarities
          this.itemSimilarity.set(`${itemIds[i]}-${itemIds[j]}`, similarity);
        }
      }
    }
  }

  // Utility Methods
  findSimilarUsers(userId, limit) {
    const similarities = [];
    
    for (const [key, similarity] of this.userSimilarity) {
      const [user1, user2] = key.split('-');
      if (user1 === userId || user2 === userId) {
        const otherUser = user1 === userId ? user2 : user1;
        similarities.push([otherUser, similarity]);
      }
    }
    
    return similarities
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit);
  }

  findSimilarItems(itemId, limit) {
    const similarities = [];
    
    for (const [key, similarity] of this.itemSimilarity) {
      const [item1, item2] = key.split('-');
      if (item1 === itemId || item2 === itemId) {
        const otherItem = item1 === itemId ? item2 : item1;
        similarities.push([otherItem, similarity]);
      }
    }
    
    return similarities
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit);
  }

  getUserItems(userId) {
    const items = new Map();
    
    for (const interaction of this.interactions.values()) {
      if (interaction.userId === userId) {
        const currentScore = items.get(interaction.itemId) || 0;
        items.set(interaction.itemId, currentScore + interaction.score);
      }
    }
    
    return items;
  }

  getItemUsers(itemId) {
    const users = new Map();
    
    for (const interaction of this.interactions.values()) {
      if (interaction.itemId === itemId) {
        const currentScore = users.get(interaction.userId) || 0;
        users.set(interaction.userId, currentScore + interaction.score);
      }
    }
    
    return users;
  }

  hasUserInteracted(userId, itemId) {
    for (const interaction of this.interactions.values()) {
      if (interaction.userId === userId && interaction.itemId === itemId) {
        return true;
      }
    }
    return false;
  }

  normalizeScore(score) {
    return Math.min(1, Math.max(0, score));
  }

  generateID() {
    return uuidv4();
  }
}

// Algorithm Classes
class CollaborativeFiltering {
  constructor(recommendationService) {
    this.service = recommendationService;
  }
  
  async generate(userId, options) {
    return await this.service.generateCollaborativeRecommendations(userId, options);
  }
}

class ContentBasedFiltering {
  constructor(recommendationService) {
    this.service = recommendationService;
  }
  
  async generate(userId, options) {
    return await this.service.generateContentBasedRecommendations(userId, options);
  }
}

class HybridRecommendation {
  constructor(recommendationService) {
    this.service = recommendationService;
  }
  
  async generate(userId, options) {
    return await this.service.generateHybridRecommendations(userId, options);
  }
}
```

## Key Features

### Recommendation Algorithms
- **Collaborative Filtering**: User-based and item-based recommendations
- **Content-Based Filtering**: Recommendations based on item features
- **Hybrid Approach**: Combines multiple algorithms for better accuracy
- **Real-time Updates**: Recommendations update based on user interactions

### Performance Optimization
- **Caching Strategy**: Multi-level caching for fast recommendations
- **Background Processing**: Asynchronous similarity calculations
- **Batch Processing**: Efficient recommendation generation for active users
- **Similarity Precomputation**: Pre-calculate user and item similarities

### Scalability Features
- **Distributed Architecture**: Support for multiple recommendation servers
- **Load Balancing**: Distribute recommendation requests across servers
- **Database Optimization**: Efficient storage and retrieval of user-item data
- **Memory Management**: Optimized data structures for large-scale data

## Extension Ideas

### Advanced Features
1. **Deep Learning Models**: Neural collaborative filtering
2. **Real-time Streaming**: Apache Kafka for real-time interaction processing
3. **A/B Testing Framework**: Test different recommendation algorithms
4. **Cold Start Solutions**: Handle new users and items
5. **Multi-armed Bandit**: Optimize recommendation exploration vs exploitation

### Enterprise Features
1. **Multi-tenant Support**: Isolated recommendations per tenant
2. **Advanced Analytics**: Recommendation performance metrics
3. **Personalization Engine**: Dynamic algorithm selection per user
4. **Content Moderation**: Filter inappropriate recommendations
5. **Compliance**: GDPR and privacy compliance features
