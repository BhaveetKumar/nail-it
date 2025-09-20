# API Design

## Table of Contents

1. [Overview](#overview)
2. [RESTful API Design](#restful-api-design)
3. [GraphQL APIs](#graphql-apis)
4. [API Documentation](#api-documentation)
5. [API Testing](#api-testing)
6. [API Security](#api-security)
7. [Performance & Optimization](#performance--optimization)
8. [Implementations](#implementations)
9. [Follow-up Questions](#follow-up-questions)
10. [Sources](#sources)
11. [Projects](#projects)

## Overview

### Learning Objectives

- Design RESTful APIs following best practices
- Implement GraphQL APIs with proper schema design
- Create comprehensive API documentation
- Implement API testing strategies
- Apply security best practices to APIs
- Optimize API performance and scalability

### What is API Design?

API Design covers the principles, patterns, and best practices for creating well-designed, secure, and scalable APIs that serve as the foundation for modern applications.

## RESTful API Design

### 1. REST Principles

#### Resource-Based URLs
```javascript
// Good RESTful design
GET    /api/users              // Get all users
GET    /api/users/123          // Get user by ID
POST   /api/users              // Create new user
PUT    /api/users/123          // Update user (full)
PATCH  /api/users/123          // Update user (partial)
DELETE /api/users/123          // Delete user

// Nested resources
GET    /api/users/123/posts    // Get user's posts
POST   /api/users/123/posts    // Create post for user
GET    /api/posts/456/comments // Get post comments
```

#### HTTP Status Codes
```javascript
// Success responses
200 OK          // Successful GET, PUT, PATCH
201 Created     // Successful POST
204 No Content  // Successful DELETE

// Client errors
400 Bad Request     // Invalid request data
401 Unauthorized    // Authentication required
403 Forbidden       // Insufficient permissions
404 Not Found       // Resource not found
409 Conflict        // Resource conflict
422 Unprocessable Entity // Validation errors

// Server errors
500 Internal Server Error // Server error
503 Service Unavailable   // Service down
```

### 2. API Implementation

#### Express.js REST API
```javascript
// routes/users.js
const express = require('express');
const router = express.Router();
const { body, param, query, validationResult } = require('express-validator');

// Validation middleware
const validateRequest = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({
      error: 'Validation failed',
      details: errors.array()
    });
  }
  next();
};

// GET /api/users
router.get('/', [
  query('page').optional().isInt({ min: 1 }),
  query('limit').optional().isInt({ min: 1, max: 100 }),
  query('search').optional().isLength({ min: 1, max: 100 })
], validateRequest, async (req, res) => {
  try {
    const { page = 1, limit = 10, search } = req.query;
    const offset = (page - 1) * limit;
    
    const query = search ? { name: { $regex: search, $options: 'i' } } : {};
    const users = await User.find(query)
      .select('-password')
      .skip(offset)
      .limit(parseInt(limit))
      .sort({ createdAt: -1 });
    
    const total = await User.countDocuments(query);
    
    res.json({
      data: users,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total,
        pages: Math.ceil(total / limit)
      }
    });
  } catch (error) {
    res.status(500).json({ error: 'Internal server error' });
  }
});

// POST /api/users
router.post('/', [
  body('email').isEmail().normalizeEmail(),
  body('password').isLength({ min: 8 }),
  body('name').trim().isLength({ min: 2, max: 50 })
], validateRequest, async (req, res) => {
  try {
    const { email, password, name } = req.body;
    
    // Check if user exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(409).json({ error: 'User already exists' });
    }
    
    // Create user
    const user = new User({ email, password, name });
    await user.save();
    
    res.status(201).json({
      data: {
        id: user._id,
        email: user.email,
        name: user.name,
        createdAt: user.createdAt
      }
    });
  } catch (error) {
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router;
```

#### Go Gin REST API
```go
// handlers/user.go
package handlers

import (
    "net/http"
    "strconv"
    "time"

    "github.com/gin-gonic/gin"
    "github.com/go-playground/validator/v10"
)

type UserHandler struct {
    userService UserService
    validator   *validator.Validate
}

type CreateUserRequest struct {
    Email    string `json:"email" binding:"required,email"`
    Password string `json:"password" binding:"required,min=8"`
    Name     string `json:"name" binding:"required,min=2,max=50"`
}

type UpdateUserRequest struct {
    Name  string `json:"name" binding:"omitempty,min=2,max=50"`
    Email string `json:"email" binding:"omitempty,email"`
}

func (h *UserHandler) GetUsers(c *gin.Context) {
    page, _ := strconv.Atoi(c.DefaultQuery("page", "1"))
    limit, _ := strconv.Atoi(c.DefaultQuery("limit", "10"))
    search := c.Query("search")
    
    users, total, err := h.userService.GetUsers(page, limit, search)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch users"})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{
        "data": users,
        "pagination": gin.H{
            "page":  page,
            "limit": limit,
            "total": total,
            "pages": (total + int64(limit) - 1) / int64(limit),
        },
    })
}

func (h *UserHandler) CreateUser(c *gin.Context) {
    var req CreateUserRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    
    user, err := h.userService.CreateUser(req.Email, req.Password, req.Name)
    if err != nil {
        if err == ErrUserExists {
            c.JSON(http.StatusConflict, gin.H{"error": "User already exists"})
            return
        }
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create user"})
        return
    }
    
    c.JSON(http.StatusCreated, gin.H{
        "data": gin.H{
            "id":        user.ID,
            "email":     user.Email,
            "name":      user.Name,
            "createdAt": user.CreatedAt,
        },
    })
}
```

## GraphQL APIs

### 1. Schema Design

#### GraphQL Schema Definition
```graphql
# schema.graphql
type User {
  id: ID!
  email: String!
  name: String!
  posts: [Post!]!
  createdAt: DateTime!
  updatedAt: DateTime!
}

type Post {
  id: ID!
  title: String!
  content: String!
  author: User!
  comments: [Comment!]!
  createdAt: DateTime!
  updatedAt: DateTime!
}

type Comment {
  id: ID!
  content: String!
  author: User!
  post: Post!
  createdAt: DateTime!
}

type Query {
  users(first: Int, after: String, search: String): UserConnection!
  user(id: ID!): User
  posts(first: Int, after: String, authorId: ID): PostConnection!
  post(id: ID!): Post
}

type Mutation {
  createUser(input: CreateUserInput!): User!
  updateUser(id: ID!, input: UpdateUserInput!): User!
  deleteUser(id: ID!): Boolean!
  createPost(input: CreatePostInput!): Post!
  createComment(input: CreateCommentInput!): Comment!
}

input CreateUserInput {
  email: String!
  password: String!
  name: String!
}

input UpdateUserInput {
  name: String
  email: String
}

input CreatePostInput {
  title: String!
  content: String!
  authorId: ID!
}

input CreateCommentInput {
  content: String!
  authorId: ID!
  postId: ID!
}

type UserConnection {
  edges: [UserEdge!]!
  pageInfo: PageInfo!
}

type UserEdge {
  node: User!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}
```

### 2. GraphQL Implementation

#### Apollo Server Setup
```javascript
// server.js
const { ApolloServer } = require('apollo-server-express');
const { makeExecutableSchema } = require('@graphql-tools/schema');
const { applyMiddleware } = require('graphql-middleware');

const typeDefs = require('./schema');
const resolvers = require('./resolvers');
const permissions = require('./permissions');

const schema = makeExecutableSchema({ typeDefs, resolvers });
const schemaWithMiddleware = applyMiddleware(schema, permissions);

const server = new ApolloServer({
  schema: schemaWithMiddleware,
  context: ({ req }) => ({
    user: req.user,
    dataSources: {
      userAPI: new UserAPI(),
      postAPI: new PostAPI(),
    },
  }),
  formatError: (error) => {
    console.error(error);
    return {
      message: error.message,
      code: error.extensions?.code,
      path: error.path,
    };
  },
});

// resolvers/user.js
const resolvers = {
  Query: {
    users: async (_, { first = 10, after, search }, { dataSources }) => {
      return dataSources.userAPI.getUsers({ first, after, search });
    },
    user: async (_, { id }, { dataSources }) => {
      return dataSources.userAPI.getUserById(id);
    },
  },
  Mutation: {
    createUser: async (_, { input }, { dataSources }) => {
      return dataSources.userAPI.createUser(input);
    },
    updateUser: async (_, { id, input }, { dataSources, user }) => {
      if (user.id !== id && user.role !== 'admin') {
        throw new Error('Unauthorized');
      }
      return dataSources.userAPI.updateUser(id, input);
    },
  },
  User: {
    posts: async (parent, { first = 10, after }, { dataSources }) => {
      return dataSources.postAPI.getPostsByAuthor(parent.id, { first, after });
    },
  },
};
```

## API Documentation

### 1. OpenAPI Specification

#### OpenAPI 3.0 Definition
```yaml
# openapi.yaml
openapi: 3.0.0
info:
  title: User Management API
  description: API for managing users and their data
  version: 1.0.0
  contact:
    name: API Support
    email: support@example.com
servers:
  - url: https://api.example.com/v1
    description: Production server
  - url: https://staging-api.example.com/v1
    description: Staging server

paths:
  /users:
    get:
      summary: Get all users
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            minimum: 1
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            minimum: 1
            maximum: 100
            default: 10
        - name: search
          in: query
          schema:
            type: string
            maxLength: 100
      responses:
        '200':
          description: List of users
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/User'
                  pagination:
                    $ref: '#/components/schemas/Pagination'
        '400':
          $ref: '#/components/responses/BadRequest'
        '500':
          $ref: '#/components/responses/InternalError'
    
    post:
      summary: Create a new user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
      responses:
        '201':
          description: User created successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    $ref: '#/components/schemas/User'
        '400':
          $ref: '#/components/responses/BadRequest'
        '409':
          $ref: '#/components/responses/Conflict'

components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: string
          format: uuid
        email:
          type: string
          format: email
        name:
          type: string
        createdAt:
          type: string
          format: date-time
        updatedAt:
          type: string
          format: date-time
      required:
        - id
        - email
        - name
        - createdAt
        - updatedAt
    
    CreateUserRequest:
      type: object
      properties:
        email:
          type: string
          format: email
        password:
          type: string
          minLength: 8
        name:
          type: string
          minLength: 2
          maxLength: 50
      required:
        - email
        - password
        - name
    
    Pagination:
      type: object
      properties:
        page:
          type: integer
        limit:
          type: integer
        total:
          type: integer
        pages:
          type: integer
  
  responses:
    BadRequest:
      description: Bad request
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
    
    Conflict:
      description: Resource conflict
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
    
    InternalError:
      description: Internal server error
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
```

## API Testing

### 1. Unit Testing

#### Jest API Tests
```javascript
// tests/api/users.test.js
const request = require('supertest');
const app = require('../../app');
const User = require('../../models/User');

describe('Users API', () => {
  beforeEach(async () => {
    await User.deleteMany({});
  });

  describe('GET /api/users', () => {
    it('should return empty array when no users exist', async () => {
      const response = await request(app)
        .get('/api/users')
        .expect(200);
      
      expect(response.body.data).toEqual([]);
      expect(response.body.pagination.total).toBe(0);
    });

    it('should return users with pagination', async () => {
      // Create test users
      await User.create([
        { email: 'user1@test.com', name: 'User 1', password: 'password123' },
        { email: 'user2@test.com', name: 'User 2', password: 'password123' },
      ]);

      const response = await request(app)
        .get('/api/users?page=1&limit=1')
        .expect(200);
      
      expect(response.body.data).toHaveLength(1);
      expect(response.body.pagination.total).toBe(2);
      expect(response.body.pagination.pages).toBe(2);
    });
  });

  describe('POST /api/users', () => {
    it('should create a new user', async () => {
      const userData = {
        email: 'test@example.com',
        password: 'password123',
        name: 'Test User'
      };

      const response = await request(app)
        .post('/api/users')
        .send(userData)
        .expect(201);
      
      expect(response.body.data.email).toBe(userData.email);
      expect(response.body.data.name).toBe(userData.name);
      expect(response.body.data.password).toBeUndefined();
    });

    it('should return 400 for invalid data', async () => {
      const invalidData = {
        email: 'invalid-email',
        password: '123',
        name: ''
      };

      const response = await request(app)
        .post('/api/users')
        .send(invalidData)
        .expect(400);
      
      expect(response.body.error).toBe('Validation failed');
    });
  });
});
```

### 2. Integration Testing

#### API Integration Tests
```javascript
// tests/integration/api.test.js
const request = require('supertest');
const app = require('../../app');
const { setupTestDB, cleanupTestDB } = require('../helpers/db');

describe('API Integration Tests', () => {
  beforeAll(async () => {
    await setupTestDB();
  });

  afterAll(async () => {
    await cleanupTestDB();
  });

  describe('User Workflow', () => {
    let authToken;
    let userId;

    it('should complete user registration and login flow', async () => {
      // Register user
      const registerResponse = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'test@example.com',
          password: 'password123',
          name: 'Test User'
        })
        .expect(201);
      
      authToken = registerResponse.body.token;
      userId = registerResponse.body.user.id;

      // Login user
      const loginResponse = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'test@example.com',
          password: 'password123'
        })
        .expect(200);
      
      expect(loginResponse.body.token).toBeDefined();
    });

    it('should allow authenticated user to access protected routes', async () => {
      const response = await request(app)
        .get('/api/users')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);
      
      expect(response.body.data).toBeDefined();
    });
  });
});
```

## Follow-up Questions

### 1. RESTful Design
**Q: What's the difference between PUT and PATCH HTTP methods?**
A: PUT replaces the entire resource (idempotent), while PATCH applies partial modifications to a resource.

### 2. GraphQL
**Q: What are the advantages of GraphQL over REST?**
A: GraphQL provides a single endpoint, allows clients to request exactly the data they need, reduces over-fetching, and provides strong typing and introspection.

### 3. API Security
**Q: How do you prevent API abuse and rate limiting?**
A: Implement rate limiting, authentication, input validation, CORS policies, and monitoring to prevent abuse and ensure API security.

## Sources

### Books
- **RESTful Web APIs** by Leonard Richardson
- **GraphQL in Action** by Samer Buna
- **API Design Patterns** by JJ Geewax

### Online Resources
- **REST API Tutorial** - REST best practices
- **GraphQL Documentation** - Official GraphQL guides
- **OpenAPI Specification** - API documentation standards

## Projects

### 1. RESTful API Service
**Objective**: Build a comprehensive REST API
**Requirements**: CRUD operations, authentication, documentation
**Deliverables**: Complete API with OpenAPI documentation

### 2. GraphQL API
**Objective**: Create a GraphQL API with advanced features
**Requirements**: Schema design, resolvers, subscriptions
**Deliverables**: GraphQL API with real-time capabilities

### 3. API Testing Suite
**Objective**: Develop comprehensive API testing
**Requirements**: Unit tests, integration tests, load tests
**Deliverables**: Complete testing framework with CI/CD integration

---

**Next**: [System Design Basics](../../../README.md) | **Previous**: [Web Development](../../../README.md) | **Up**: [Phase 1](README.md)

