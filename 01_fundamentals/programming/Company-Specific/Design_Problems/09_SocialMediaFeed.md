---
# Auto-generated front matter
Title: 09 Socialmediafeed
LastUpdated: 2025-11-06T20:45:58.772053
Tags: []
Status: draft
---

# 09. Social Media Feed - News Feed System

## Title & Summary
Design and implement a social media news feed system using Node.js that handles real-time updates, personalized content ranking, and supports millions of users with high availability.

## Problem Statement

Build a comprehensive social media feed system that:

1. **Feed Generation**: Create personalized news feeds for users
2. **Real-time Updates**: Push new content to followers instantly
3. **Content Ranking**: Algorithm-based content prioritization
4. **Social Features**: Likes, comments, shares, and follows
5. **Content Types**: Text, images, videos, and links
6. **Scalability**: Handle millions of users and posts

## Requirements & Constraints

### Functional Requirements
- Generate personalized news feeds
- Real-time content updates
- Content ranking and filtering
- Social interactions (like, comment, share)
- User following and unfollowing
- Content moderation and reporting
- Trending topics and hashtags

### Non-Functional Requirements
- **Latency**: < 200ms for feed generation
- **Throughput**: 100,000+ posts per minute
- **Availability**: 99.9% uptime
- **Scalability**: Support 100M+ users
- **Consistency**: Eventual consistency for social features
- **Storage**: Efficient storage and retrieval

## API / Interfaces

### REST Endpoints

```javascript
// Feed Management
GET    /api/feed/{userId}
POST   /api/feed/refresh
GET    /api/feed/trending

// Content Management
POST   /api/posts
GET    /api/posts/{postId}
PUT    /api/posts/{postId}
DELETE /api/posts/{postId}
GET    /api/posts/user/{userId}

// Social Interactions
POST   /api/posts/{postId}/like
POST   /api/posts/{postId}/comment
POST   /api/posts/{postId}/share
GET    /api/posts/{postId}/comments

// User Management
POST   /api/users/{userId}/follow
DELETE /api/users/{userId}/follow
GET    /api/users/{userId}/followers
GET    /api/users/{userId}/following
```

### Request/Response Examples

```json
// Create Post
POST /api/posts
{
  "userId": "user_123",
  "content": "Just had an amazing day at the beach! 🏖️",
  "type": "text",
  "media": [],
  "hashtags": ["beach", "vacation"],
  "mentions": ["@friend_456"]
}

// Response
{
  "success": true,
  "data": {
    "postId": "post_789",
    "userId": "user_123",
    "content": "Just had an amazing day at the beach! 🏖️",
    "type": "text",
    "hashtags": ["beach", "vacation"],
    "mentions": ["@friend_456"],
    "likes": 0,
    "comments": 0,
    "shares": 0,
    "createdAt": "2024-01-15T10:30:00Z"
  }
}

// Get Feed
GET /api/feed/user_123?limit=20&offset=0

// Response
{
  "success": true,
  "data": {
    "posts": [
      {
        "postId": "post_789",
        "userId": "user_456",
        "username": "john_doe",
        "content": "Amazing sunset today!",
        "type": "image",
        "media": ["https://cdn.example.com/image1.jpg"],
        "likes": 25,
        "comments": 5,
        "shares": 2,
        "liked": false,
        "createdAt": "2024-01-15T09:15:00Z",
        "score": 0.85
      }
    ],
    "pagination": {
      "limit": 20,
      "offset": 0,
      "hasMore": true
    }
  }
}
```

## Data Model

### Core Entities

```javascript
// Post Entity
class Post {
  constructor(userId, content, type) {
    this.id = this.generateID();
    this.userId = userId;
    this.content = content;
    this.type = type; // 'text', 'image', 'video', 'link'
    this.media = [];
    this.hashtags = [];
    this.mentions = [];
    this.likes = 0;
    this.comments = 0;
    this.shares = 0;
    this.score = 0;
    this.isActive = true;
    this.createdAt = new Date();
    this.updatedAt = new Date();
  }
}

// User Entity
class User {
  constructor(username, email) {
    this.id = this.generateID();
    this.username = username;
    this.email = email;
    this.displayName = "";
    this.bio = "";
    this.avatar = "";
    this.followers = 0;
    this.following = 0;
    this.posts = 0;
    this.isActive = true;
    this.createdAt = new Date();
    this.updatedAt = new Date();
  }
}

// Follow Relationship Entity
class Follow {
  constructor(followerId, followingId) {
    this.id = this.generateID();
    this.followerId = followerId;
    this.followingId = followingId;
    this.createdAt = new Date();
  }
}

// Like Entity
class Like {
  constructor(userId, postId) {
    this.id = this.generateID();
    this.userId = userId;
    this.postId = postId;
    this.createdAt = new Date();
  }
}

// Comment Entity
class Comment {
  constructor(userId, postId, content) {
    this.id = this.generateID();
    this.userId = userId;
    this.postId = postId;
    this.content = content;
    this.likes = 0;
    this.replies = 0;
    this.isActive = true;
    this.createdAt = new Date();
    this.updatedAt = new Date();
  }
}

// Feed Cache Entity
class FeedCache {
  constructor(userId, posts) {
    this.userId = userId;
    this.posts = posts;
    this.generatedAt = new Date();
    this.expiresAt = new Date(Date.now() + 300000); // 5 minutes
  }
}
```

## Approach Overview

### Simple Solution (MVP)
1. In-memory storage with arrays
2. Basic feed generation
3. Simple social interactions
4. No ranking or personalization

### Production-Ready Design
1. **Distributed Architecture**: Multiple service instances
2. **Feed Ranking**: ML-based content ranking algorithm
3. **Real-time Updates**: WebSocket for live feed updates
4. **Caching Strategy**: Multi-level caching for performance
5. **Content Moderation**: AI-based content filtering
6. **Analytics**: User engagement tracking

## Detailed Design

### Core Service Implementation

```javascript
const EventEmitter = require("events");
const { v4: uuidv4 } = require("uuid");

class SocialMediaFeedService extends EventEmitter {
  constructor() {
    super();
    this.posts = new Map();
    this.users = new Map();
    this.follows = new Map();
    this.likes = new Map();
    this.comments = new Map();
    this.feedCache = new Map();
    this.trendingHashtags = new Map();
    this.userEngagement = new Map();
    
    // Start background tasks
    this.startFeedGenerator();
    this.startTrendingCalculator();
    this.startCacheCleanup();
  }

  // Post Management
  async createPost(postData) {
    try {
      const post = new Post(
        postData.userId,
        postData.content,
        postData.type
      );
      
      // Set additional properties
      if (postData.media) post.media = postData.media;
      if (postData.hashtags) post.hashtags = postData.hashtags;
      if (postData.mentions) post.mentions = postData.mentions;
      
      // Store post
      this.posts.set(post.id, post);
      
      // Update user post count
      const user = this.users.get(post.userId);
      if (user) {
        user.posts++;
        user.updatedAt = new Date();
      }
      
      // Update trending hashtags
      this.updateTrendingHashtags(post.hashtags);
      
      // Invalidate follower feeds
      await this.invalidateFollowerFeeds(post.userId);
      
      this.emit("postCreated", post);
      
      return post;
      
    } catch (error) {
      console.error("Post creation error:", error);
      throw error;
    }
  }

  // Feed Generation
  async generateFeed(userId, options = {}) {
    try {
      const { limit = 20, offset = 0 } = options;
      
      // Check cache first
      const cachedFeed = this.getCachedFeed(userId);
      if (cachedFeed && cachedFeed.expiresAt > new Date()) {
        return this.paginateFeed(cachedFeed.posts, limit, offset);
      }
      
      // Get following users
      const following = this.getFollowing(userId);
      
      // Get posts from following users
      const posts = await this.getPostsFromUsers(following);
      
      // Rank posts
      const rankedPosts = await this.rankPosts(posts, userId);
      
      // Cache feed
      this.cacheFeed(userId, rankedPosts);
      
      return this.paginateFeed(rankedPosts, limit, offset);
      
    } catch (error) {
      console.error("Feed generation error:", error);
      throw error;
    }
  }

  // Post Ranking Algorithm
  async rankPosts(posts, userId) {
    const userEngagement = this.getUserEngagement(userId);
    
    return posts.map(post => {
      let score = 0;
      
      // Recency score (newer posts get higher score)
      const ageInHours = (new Date() - post.createdAt) / (1000 * 60 * 60);
      score += Math.max(0, 1 - (ageInHours / 24)) * 0.3;
      
      // Engagement score
      const engagementScore = (post.likes + post.comments * 2 + post.shares * 3) / 100;
      score += Math.min(engagementScore, 1) * 0.4;
      
      // User relationship score
      const relationshipScore = this.calculateRelationshipScore(post.userId, userId);
      score += relationshipScore * 0.2;
      
      // Content type preference
      const typePreference = this.getContentTypePreference(userId, post.type);
      score += typePreference * 0.1;
      
      post.score = score;
      return post;
    }).sort((a, b) => b.score - a.score);
  }

  // Social Interactions
  async likePost(userId, postId) {
    try {
      const likeKey = `${userId}_${postId}`;
      
      if (this.likes.has(likeKey)) {
        throw new Error("Post already liked");
      }
      
      const like = new Like(userId, postId);
      this.likes.set(likeKey, like);
      
      // Update post like count
      const post = this.posts.get(postId);
      if (post) {
        post.likes++;
        post.updatedAt = new Date();
      }
      
      // Update user engagement
      this.updateUserEngagement(userId, 'like');
      
      this.emit("postLiked", { userId, postId, like });
      
      return like;
      
    } catch (error) {
      console.error("Like post error:", error);
      throw error;
    }
  }

  async commentOnPost(userId, postId, content) {
    try {
      const comment = new Comment(userId, postId, content);
      this.comments.set(comment.id, comment);
      
      // Update post comment count
      const post = this.posts.get(postId);
      if (post) {
        post.comments++;
        post.updatedAt = new Date();
      }
      
      // Update user engagement
      this.updateUserEngagement(userId, 'comment');
      
      this.emit("postCommented", { userId, postId, comment });
      
      return comment;
      
    } catch (error) {
      console.error("Comment post error:", error);
      throw error;
    }
  }

  // Follow Management
  async followUser(followerId, followingId) {
    try {
      if (followerId === followingId) {
        throw new Error("Cannot follow yourself");
      }
      
      const followKey = `${followerId}_${followingId}`;
      
      if (this.follows.has(followKey)) {
        throw new Error("Already following this user");
      }
      
      const follow = new Follow(followerId, followingId);
      this.follows.set(followKey, follow);
      
      // Update user counts
      const follower = this.users.get(followerId);
      const following = this.users.get(followingId);
      
      if (follower) {
        follower.following++;
        follower.updatedAt = new Date();
      }
      
      if (following) {
        following.followers++;
        following.updatedAt = new Date();
      }
      
      // Invalidate follower's feed cache
      this.invalidateFeedCache(followerId);
      
      this.emit("userFollowed", { followerId, followingId, follow });
      
      return follow;
      
    } catch (error) {
      console.error("Follow user error:", error);
      throw error;
    }
  }

  // Trending Topics
  async getTrendingHashtags(limit = 10) {
    const trending = Array.from(this.trendingHashtags.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit)
      .map(([hashtag, count]) => ({ hashtag, count }));
    
    return trending;
  }

  // Background Tasks
  startFeedGenerator() {
    setInterval(() => {
      this.generateTrendingFeeds();
    }, 60000); // Run every minute
  }

  async generateTrendingFeeds() {
    // Generate trending content for users who haven't been active
    const inactiveUsers = this.getInactiveUsers();
    
    for (const userId of inactiveUsers) {
      try {
        await this.generateFeed(userId);
      } catch (error) {
        console.error(`Failed to generate feed for user ${userId}:`, error);
      }
    }
  }

  startTrendingCalculator() {
    setInterval(() => {
      this.calculateTrendingHashtags();
    }, 300000); // Run every 5 minutes
  }

  calculateTrendingHashtags() {
    // Reset trending hashtags
    this.trendingHashtags.clear();
    
    // Count hashtag usage in recent posts
    const recentPosts = this.getRecentPosts(24); // Last 24 hours
    
    for (const post of recentPosts) {
      for (const hashtag of post.hashtags) {
        const count = this.trendingHashtags.get(hashtag) || 0;
        this.trendingHashtags.set(hashtag, count + 1);
      }
    }
    
    this.emit("trendingUpdated", this.trendingHashtags);
  }

  startCacheCleanup() {
    setInterval(() => {
      this.cleanupExpiredCache();
    }, 300000); // Run every 5 minutes
  }

  cleanupExpiredCache() {
    const now = new Date();
    const expiredFeeds = [];
    
    for (const [userId, feedCache] of this.feedCache) {
      if (feedCache.expiresAt < now) {
        expiredFeeds.push(userId);
      }
    }
    
    expiredFeeds.forEach(userId => {
      this.feedCache.delete(userId);
    });
    
    if (expiredFeeds.length > 0) {
      this.emit("cacheCleaned", expiredFeeds.length);
    }
  }

  // Utility Methods
  getFollowing(userId) {
    const following = [];
    
    for (const [followKey, follow] of this.follows) {
      if (follow.followerId === userId) {
        following.push(follow.followingId);
      }
    }
    
    return following;
  }

  getPostsFromUsers(userIds) {
    const posts = [];
    
    for (const post of this.posts.values()) {
      if (userIds.includes(post.userId) && post.isActive) {
        posts.push(post);
      }
    }
    
    return posts.sort((a, b) => b.createdAt - a.createdAt);
  }

  getCachedFeed(userId) {
    return this.feedCache.get(userId);
  }

  cacheFeed(userId, posts) {
    const feedCache = new FeedCache(userId, posts);
    this.feedCache.set(userId, feedCache);
  }

  invalidateFeedCache(userId) {
    this.feedCache.delete(userId);
  }

  async invalidateFollowerFeeds(userId) {
    const followers = this.getFollowers(userId);
    
    for (const followerId of followers) {
      this.invalidateFeedCache(followerId);
    }
  }

  getFollowers(userId) {
    const followers = [];
    
    for (const [followKey, follow] of this.follows) {
      if (follow.followingId === userId) {
        followers.push(follow.followerId);
      }
    }
    
    return followers;
  }

  calculateRelationshipScore(posterId, viewerId) {
    // Simple relationship scoring based on interaction history
    const interactions = this.getUserInteractions(posterId, viewerId);
    return Math.min(interactions / 10, 1); // Normalize to 0-1
  }

  getUserInteractions(userId1, userId2) {
    // Count interactions between two users
    let interactions = 0;
    
    for (const like of this.likes.values()) {
      if ((like.userId === userId1 && this.posts.get(like.postId)?.userId === userId2) ||
          (like.userId === userId2 && this.posts.get(like.postId)?.userId === userId1)) {
        interactions++;
      }
    }
    
    return interactions;
  }

  getContentTypePreference(userId, contentType) {
    const userEngagement = this.getUserEngagement(userId);
    return userEngagement.contentTypePreferences[contentType] || 0.5;
  }

  getUserEngagement(userId) {
    return this.userEngagement.get(userId) || {
      likes: 0,
      comments: 0,
      shares: 0,
      contentTypePreferences: {
        text: 0.5,
        image: 0.5,
        video: 0.5,
        link: 0.5
      }
    };
  }

  updateUserEngagement(userId, action) {
    const engagement = this.getUserEngagement(userId);
    engagement[action]++;
    this.userEngagement.set(userId, engagement);
  }

  updateTrendingHashtags(hashtags) {
    for (const hashtag of hashtags) {
      const count = this.trendingHashtags.get(hashtag) || 0;
      this.trendingHashtags.set(hashtag, count + 1);
    }
  }

  getRecentPosts(hours) {
    const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000);
    
    return Array.from(this.posts.values())
      .filter(post => post.createdAt > cutoff)
      .sort((a, b) => b.createdAt - a.createdAt);
  }

  getInactiveUsers() {
    // Return users who haven't been active in the last hour
    const cutoff = new Date(Date.now() - 60 * 60 * 1000);
    
    return Array.from(this.users.keys()).filter(userId => {
      const user = this.users.get(userId);
      return user.updatedAt < cutoff;
    });
  }

  paginateFeed(posts, limit, offset) {
    return {
      posts: posts.slice(offset, offset + limit),
      pagination: {
        limit,
        offset,
        total: posts.length,
        hasMore: offset + limit < posts.length
      }
    };
  }

  generateID() {
    return uuidv4();
  }
}
```

### Express.js API Implementation

```javascript
const express = require("express");
const cors = require("cors");
const { SocialMediaFeedService } = require("./services/SocialMediaFeedService");

class SocialMediaFeedAPI {
  constructor() {
    this.app = express();
    this.feedService = new SocialMediaFeedService();
    
    this.setupMiddleware();
    this.setupRoutes();
    this.setupEventHandlers();
  }

  setupMiddleware() {
    this.app.use(cors());
    this.app.use(express.json());
    this.app.use(express.urlencoded({ extended: true }));
    
    // Request logging
    this.app.use((req, res, next) => {
      console.log(`${req.method} ${req.path} - ${new Date().toISOString()}`);
      next();
    });
  }

  setupRoutes() {
    // Feed management
    this.app.get("/api/feed/:userId", this.getFeed.bind(this));
    this.app.post("/api/feed/refresh", this.refreshFeed.bind(this));
    this.app.get("/api/feed/trending", this.getTrendingFeed.bind(this));
    
    // Post management
    this.app.post("/api/posts", this.createPost.bind(this));
    this.app.get("/api/posts/:postId", this.getPost.bind(this));
    this.app.put("/api/posts/:postId", this.updatePost.bind(this));
    this.app.delete("/api/posts/:postId", this.deletePost.bind(this));
    this.app.get("/api/posts/user/:userId", this.getUserPosts.bind(this));
    
    // Social interactions
    this.app.post("/api/posts/:postId/like", this.likePost.bind(this));
    this.app.post("/api/posts/:postId/comment", this.commentOnPost.bind(this));
    this.app.post("/api/posts/:postId/share", this.sharePost.bind(this));
    this.app.get("/api/posts/:postId/comments", this.getPostComments.bind(this));
    
    // User management
    this.app.post("/api/users/:userId/follow", this.followUser.bind(this));
    this.app.delete("/api/users/:userId/follow", this.unfollowUser.bind(this));
    this.app.get("/api/users/:userId/followers", this.getFollowers.bind(this));
    this.app.get("/api/users/:userId/following", this.getFollowing.bind(this));
    
    // Trending
    this.app.get("/api/trending/hashtags", this.getTrendingHashtags.bind(this));
    
    // Health check
    this.app.get("/health", (req, res) => {
      res.json({
        status: "healthy",
        timestamp: new Date(),
        totalPosts: this.feedService.posts.size,
        totalUsers: this.feedService.users.size,
        totalFollows: this.feedService.follows.size
      });
    });
  }

  setupEventHandlers() {
    this.feedService.on("postCreated", (post) => {
      console.log(`Post created: ${post.id} by user ${post.userId}`);
    });
    
    this.feedService.on("postLiked", ({ userId, postId }) => {
      console.log(`Post ${postId} liked by user ${userId}`);
    });
    
    this.feedService.on("userFollowed", ({ followerId, followingId }) => {
      console.log(`User ${followerId} followed user ${followingId}`);
    });
  }

  // HTTP Handlers
  async getFeed(req, res) {
    try {
      const { userId } = req.params;
      const { limit = 20, offset = 0 } = req.query;
      
      const feed = await this.feedService.generateFeed(userId, { limit, offset });
      
      res.json({
        success: true,
        data: feed
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async createPost(req, res) {
    try {
      const post = await this.feedService.createPost(req.body);
      
      res.status(201).json({
        success: true,
        data: post
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async likePost(req, res) {
    try {
      const { postId } = req.params;
      const { userId } = req.body;
      
      const like = await this.feedService.likePost(userId, postId);
      
      res.json({
        success: true,
        data: like
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async commentOnPost(req, res) {
    try {
      const { postId } = req.params;
      const { userId, content } = req.body;
      
      const comment = await this.feedService.commentOnPost(userId, postId, content);
      
      res.status(201).json({
        success: true,
        data: comment
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async followUser(req, res) {
    try {
      const { userId } = req.params;
      const { followerId } = req.body;
      
      const follow = await this.feedService.followUser(followerId, userId);
      
      res.status(201).json({
        success: true,
        data: follow
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async getTrendingHashtags(req, res) {
    try {
      const { limit = 10 } = req.query;
      
      const trending = await this.feedService.getTrendingHashtags(limit);
      
      res.json({
        success: true,
        data: trending
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  start(port = 3000) {
    this.app.listen(port, () => {
      console.log(`Social Media Feed API server running on port ${port}`);
    });
  }
}

// Start server
if (require.main === module) {
  const api = new SocialMediaFeedAPI();
  api.start(3000);
}

module.exports = { SocialMediaFeedAPI };
```

## Key Features

### Feed Generation
- **Personalized Feeds**: Algorithm-based content ranking
- **Real-time Updates**: Live feed updates via WebSocket
- **Caching Strategy**: Multi-level caching for performance
- **Trending Content**: Popular hashtags and topics

### Social Interactions
- **Engagement Tracking**: Likes, comments, and shares
- **User Relationships**: Follow/unfollow functionality
- **Content Discovery**: Trending topics and hashtags
- **User Preferences**: Content type preferences

### Performance & Scalability
- **Efficient Ranking**: ML-based content scoring
- **Background Processing**: Async feed generation
- **Cache Management**: Smart cache invalidation
- **Load Balancing**: Distributed service architecture

## Extension Ideas

### Advanced Features
1. **Content Moderation**: AI-based content filtering
2. **Story Features**: Temporary content with expiration
3. **Live Streaming**: Real-time video content
4. **Group Features**: Communities and group feeds
5. **Advanced Analytics**: User behavior insights

### Enterprise Features
1. **Multi-tenancy**: Organization-based feeds
2. **Advanced Moderation**: Content policy enforcement
3. **Analytics Dashboard**: Comprehensive user insights
4. **API Rate Limiting**: Usage-based access control
5. **Content Recommendations**: ML-based suggestions
