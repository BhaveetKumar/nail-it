# Web Development

## Table of Contents

1. [Overview](#overview)
2. [Frontend Development](#frontend-development)
3. [Backend Development](#backend-development)
4. [Full-Stack Integration](#full-stack-integration)
5. [Web Security](#web-security)
6. [Performance Optimization](#performance-optimization)
7. [Modern Web Technologies](#modern-web-technologies)
8. [Implementations](#implementations)
9. [Follow-up Questions](#follow-up-questions)
10. [Sources](#sources)
11. [Projects](#projects)

## Overview

### Learning Objectives

- Master modern frontend frameworks (React, Vue)
- Build scalable backend services (Express, Gin)
- Integrate frontend and backend effectively
- Implement web security best practices
- Optimize web application performance
- Work with modern web technologies

### What is Web Development?

Web Development covers both frontend and backend development, including modern frameworks, APIs, security, performance optimization, and full-stack integration techniques.

## Frontend Development

### 1. React Fundamentals

#### Component Architecture
```jsx
// components/UserProfile.jsx
import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';

const UserProfile = ({ userId, onUpdate }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchUser = async () => {
      try {
        setLoading(true);
        const response = await fetch(`/api/users/${userId}`);
        if (!response.ok) {
          throw new Error('Failed to fetch user');
        }
        const userData = await response.json();
        setUser(userData);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchUser();
  }, [userId]);

  const handleUpdate = async (updatedData) => {
    try {
      const response = await fetch(`/api/users/${userId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updatedData),
      });

      if (!response.ok) {
        throw new Error('Failed to update user');
      }

      const updatedUser = await response.json();
      setUser(updatedUser);
      onUpdate?.(updatedUser);
    } catch (err) {
      setError(err.message);
    }
  };

  if (loading) return <div className="loading">Loading...</div>;
  if (error) return <div className="error">Error: {error}</div>;
  if (!user) return <div className="not-found">User not found</div>;

  return (
    <div className="user-profile">
      <h2>{user.name}</h2>
      <p>Email: {user.email}</p>
      <p>Role: {user.role}</p>
      <button onClick={() => handleUpdate({ role: 'admin' })}>
        Make Admin
      </button>
    </div>
  );
};

UserProfile.propTypes = {
  userId: PropTypes.string.isRequired,
  onUpdate: PropTypes.func,
};

export default UserProfile;
```

#### State Management with Redux
```jsx
// store/userSlice.js
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';

export const fetchUsers = createAsyncThunk(
  'users/fetchUsers',
  async (_, { rejectWithValue }) => {
    try {
      const response = await fetch('/api/users');
      if (!response.ok) {
        throw new Error('Failed to fetch users');
      }
      return await response.json();
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const createUser = createAsyncThunk(
  'users/createUser',
  async (userData, { rejectWithValue }) => {
    try {
      const response = await fetch('/api/users', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });
      if (!response.ok) {
        throw new Error('Failed to create user');
      }
      return await response.json();
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

const userSlice = createSlice({
  name: 'users',
  initialState: {
    users: [],
    loading: false,
    error: null,
  },
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchUsers.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchUsers.fulfilled, (state, action) => {
        state.loading = false;
        state.users = action.payload;
      })
      .addCase(fetchUsers.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })
      .addCase(createUser.fulfilled, (state, action) => {
        state.users.push(action.payload);
      });
  },
});

export const { clearError } = userSlice.actions;
export default userSlice.reducer;
```

### 2. Vue.js Fundamentals

#### Composition API
```vue
<!-- components/ProductList.vue -->
<template>
  <div class="product-list">
    <div class="filters">
      <input
        v-model="searchQuery"
        placeholder="Search products..."
        class="search-input"
      />
      <select v-model="selectedCategory" class="category-select">
        <option value="">All Categories</option>
        <option
          v-for="category in categories"
          :key="category.id"
          :value="category.id"
        >
          {{ category.name }}
        </option>
      </select>
    </div>
    
    <div class="products-grid">
      <ProductCard
        v-for="product in filteredProducts"
        :key="product.id"
        :product="product"
        @add-to-cart="handleAddToCart"
      />
    </div>
    
    <div v-if="loading" class="loading">Loading products...</div>
    <div v-if="error" class="error">{{ error }}</div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue';
import ProductCard from './ProductCard.vue';

const props = defineProps({
  initialProducts: {
    type: Array,
    default: () => []
  }
});

const emit = defineEmits(['add-to-cart']);

const products = ref(props.initialProducts);
const categories = ref([]);
const searchQuery = ref('');
const selectedCategory = ref('');
const loading = ref(false);
const error = ref(null);

const filteredProducts = computed(() => {
  return products.value.filter(product => {
    const matchesSearch = product.name
      .toLowerCase()
      .includes(searchQuery.value.toLowerCase());
    const matchesCategory = !selectedCategory.value || 
      product.categoryId === selectedCategory.value;
    return matchesSearch && matchesCategory;
  });
});

const fetchProducts = async () => {
  try {
    loading.value = true;
    const response = await fetch('/api/products');
    if (!response.ok) {
      throw new Error('Failed to fetch products');
    }
    products.value = await response.json();
  } catch (err) {
    error.value = err.message;
  } finally {
    loading.value = false;
  }
};

const fetchCategories = async () => {
  try {
    const response = await fetch('/api/categories');
    if (!response.ok) {
      throw new Error('Failed to fetch categories');
    }
    categories.value = await response.json();
  } catch (err) {
    console.error('Error fetching categories:', err);
  }
};

const handleAddToCart = (product) => {
  emit('add-to-cart', product);
};

// Watch for changes in search query
watch(searchQuery, (newQuery) => {
  console.log('Search query changed:', newQuery);
});

onMounted(() => {
  fetchProducts();
  fetchCategories();
});
</script>

<style scoped>
.product-list {
  padding: 20px;
}

.filters {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.search-input, .category-select {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.products-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 20px;
}

.loading, .error {
  text-align: center;
  padding: 20px;
}

.error {
  color: red;
}
</style>
```

## Backend Development

### 1. Express.js API

#### RESTful API Implementation
```javascript
// server.js
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const { body, validationResult } = require('express-validator');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP'
});
app.use('/api/', limiter);

// Authentication middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }

  jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
    if (err) {
      return res.status(403).json({ error: 'Invalid token' });
    }
    req.user = user;
    next();
  });
};

// Validation middleware
const validateRequest = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  next();
};

// Routes
app.post('/api/auth/register', [
  body('email').isEmail().normalizeEmail(),
  body('password').isLength({ min: 6 }),
  body('name').trim().isLength({ min: 2 })
], validateRequest, async (req, res) => {
  try {
    const { email, password, name } = req.body;
    
    // Check if user exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ error: 'User already exists' });
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 12);
    
    // Create user
    const user = new User({
      email,
      password: hashedPassword,
      name
    });
    
    await user.save();
    
    // Generate JWT
    const token = jwt.sign(
      { userId: user._id, email: user.email },
      process.env.JWT_SECRET,
      { expiresIn: '24h' }
    );
    
    res.status(201).json({
      message: 'User created successfully',
      token,
      user: {
        id: user._id,
        email: user.email,
        name: user.name
      }
    });
  } catch (error) {
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/users', authenticateToken, async (req, res) => {
  try {
    const { page = 1, limit = 10, search = '' } = req.query;
    const skip = (page - 1) * limit;
    
    const query = search ? { name: { $regex: search, $options: 'i' } } : {};
    
    const users = await User.find(query)
      .select('-password')
      .skip(skip)
      .limit(parseInt(limit))
      .sort({ createdAt: -1 });
    
    const total = await User.countDocuments(query);
    
    res.json({
      users,
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

app.put('/api/users/:id', authenticateToken, [
  body('name').optional().trim().isLength({ min: 2 }),
  body('email').optional().isEmail().normalizeEmail()
], validateRequest, async (req, res) => {
  try {
    const { id } = req.params;
    const updates = req.body;
    
    // Check if user exists and is authorized
    if (req.user.userId !== id && req.user.role !== 'admin') {
      return res.status(403).json({ error: 'Not authorized' });
    }
    
    const user = await User.findByIdAndUpdate(
      id,
      { $set: updates },
      { new: true, runValidators: true }
    ).select('-password');
    
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    res.json({ message: 'User updated successfully', user });
  } catch (error) {
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error(error.stack);
  res.status(500).json({ error: 'Something went wrong!' });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

### 2. Go Gin Framework

#### High-Performance API
```go
package main

import (
    "net/http"
    "strconv"
    "time"

    "github.com/gin-gonic/gin"
    "github.com/gin-contrib/cors"
    "github.com/gin-contrib/rate-limit"
    "github.com/golang-jwt/jwt/v4"
    "golang.org/x/crypto/bcrypt"
)

type User struct {
    ID        uint      `json:"id" gorm:"primaryKey"`
    Email     string    `json:"email" gorm:"uniqueIndex"`
    Name      string    `json:"name"`
    Password  string    `json:"-"`
    Role      string    `json:"role" gorm:"default:'user'"`
    CreatedAt time.Time `json:"created_at"`
    UpdatedAt time.Time `json:"updated_at"`
}

type LoginRequest struct {
    Email    string `json:"email" binding:"required,email"`
    Password string `json:"password" binding:"required,min=6"`
}

type RegisterRequest struct {
    Email    string `json:"email" binding:"required,email"`
    Password string `json:"password" binding:"required,min=6"`
    Name     string `json:"name" binding:"required,min=2"`
}

type Claims struct {
    UserID uint   `json:"user_id"`
    Email  string `json:"email"`
    Role   string `json:"role"`
    jwt.RegisteredClaims
}

type Server struct {
    router *gin.Engine
    db     *gorm.DB
    jwtKey []byte
}

func NewServer(db *gorm.DB, jwtKey string) *Server {
    r := gin.Default()
    
    // Middleware
    r.Use(cors.New(cors.Config{
        AllowOrigins:     []string{"http://localhost:3000"},
        AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
        AllowHeaders:     []string{"Origin", "Content-Type", "Authorization"},
        ExposeHeaders:    []string{"Content-Length"},
        AllowCredentials: true,
        MaxAge:           12 * time.Hour,
    }))
    
    r.Use(rateLimit.RateLimiter(rateLimit.WithRate(100, time.Minute)))
    r.Use(gin.Logger())
    r.Use(gin.Recovery())
    
    return &Server{
        router: r,
        db:     db,
        jwtKey: []byte(jwtKey),
    }
}

func (s *Server) SetupRoutes() {
    api := s.router.Group("/api")
    {
        // Auth routes
        auth := api.Group("/auth")
        {
            auth.POST("/register", s.Register)
            auth.POST("/login", s.Login)
        }
        
        // Protected routes
        protected := api.Group("/")
        protected.Use(s.AuthMiddleware())
        {
            protected.GET("/users", s.GetUsers)
            protected.GET("/users/:id", s.GetUser)
            protected.PUT("/users/:id", s.UpdateUser)
            protected.DELETE("/users/:id", s.DeleteUser)
        }
    }
}

func (s *Server) Register(c *gin.Context) {
    var req RegisterRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    
    // Check if user exists
    var existingUser User
    if err := s.db.Where("email = ?", req.Email).First(&existingUser).Error; err == nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "User already exists"})
        return
    }
    
    // Hash password
    hashedPassword, err := bcrypt.GenerateFromPassword([]byte(req.Password), bcrypt.DefaultCost)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to hash password"})
        return
    }
    
    // Create user
    user := User{
        Email:    req.Email,
        Name:     req.Name,
        Password: string(hashedPassword),
        Role:     "user",
    }
    
    if err := s.db.Create(&user).Error; err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create user"})
        return
    }
    
    // Generate JWT
    token, err := s.generateJWT(user)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to generate token"})
        return
    }
    
    c.JSON(http.StatusCreated, gin.H{
        "message": "User created successfully",
        "token":   token,
        "user": gin.H{
            "id":    user.ID,
            "email": user.Email,
            "name":  user.Name,
            "role":  user.Role,
        },
    })
}

func (s *Server) Login(c *gin.Context) {
    var req LoginRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    
    // Find user
    var user User
    if err := s.db.Where("email = ?", req.Email).First(&user).Error; err != nil {
        c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid credentials"})
        return
    }
    
    // Check password
    if err := bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(req.Password)); err != nil {
        c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid credentials"})
        return
    }
    
    // Generate JWT
    token, err := s.generateJWT(user)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to generate token"})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{
        "message": "Login successful",
        "token":   token,
        "user": gin.H{
            "id":    user.ID,
            "email": user.Email,
            "name":  user.Name,
            "role":  user.Role,
        },
    })
}

func (s *Server) GetUsers(c *gin.Context) {
    page, _ := strconv.Atoi(c.DefaultQuery("page", "1"))
    limit, _ := strconv.Atoi(c.DefaultQuery("limit", "10"))
    search := c.Query("search")
    
    offset := (page - 1) * limit
    
    var users []User
    query := s.db.Model(&User{})
    
    if search != "" {
        query = query.Where("name ILIKE ?", "%"+search+"%")
    }
    
    var total int64
    query.Count(&total)
    
    if err := query.Offset(offset).Limit(limit).Find(&users).Error; err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch users"})
        return
    }
    
    c.JSON(http.StatusOK, gin.H{
        "users": users,
        "pagination": gin.H{
            "page":  page,
            "limit": limit,
            "total": total,
            "pages": (total + int64(limit) - 1) / int64(limit),
        },
    })
}

func (s *Server) AuthMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        tokenString := c.GetHeader("Authorization")
        if tokenString == "" {
            c.JSON(http.StatusUnauthorized, gin.H{"error": "Authorization header required"})
            c.Abort()
            return
        }
        
        if len(tokenString) > 7 && tokenString[:7] == "Bearer " {
            tokenString = tokenString[7:]
        }
        
        claims := &Claims{}
        token, err := jwt.ParseWithClaims(tokenString, claims, func(token *jwt.Token) (interface{}, error) {
            return s.jwtKey, nil
        })
        
        if err != nil || !token.Valid {
            c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid token"})
            c.Abort()
            return
        }
        
        c.Set("user", claims)
        c.Next()
    }
}

func (s *Server) generateJWT(user User) (string, error) {
    claims := Claims{
        UserID: user.ID,
        Email:  user.Email,
        Role:   user.Role,
        RegisteredClaims: jwt.RegisteredClaims{
            ExpiresAt: jwt.NewNumericDate(time.Now().Add(24 * time.Hour)),
            IssuedAt:  jwt.NewNumericDate(time.Now()),
        },
    }
    
    token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    return token.SignedString(s.jwtKey)
}

func main() {
    // Database connection
    db, err := gorm.Open(postgres.Open("postgres://user:password@localhost/dbname?sslmode=disable"), &gorm.Config{})
    if err != nil {
        log.Fatal("Failed to connect to database:", err)
    }
    
    // Auto migrate
    db.AutoMigrate(&User{})
    
    // Create server
    server := NewServer(db, "your-secret-key")
    server.SetupRoutes()
    
    // Start server
    server.router.Run(":8080")
}
```

## Full-Stack Integration

### 1. API Integration

#### Frontend API Service
```javascript
// services/api.js
class ApiService {
  constructor(baseURL = '/api') {
    this.baseURL = baseURL;
    this.token = localStorage.getItem('token');
  }

  setToken(token) {
    this.token = token;
    localStorage.setItem('token', token);
  }

  removeToken() {
    this.token = null;
    localStorage.removeItem('token');
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    if (this.token) {
      config.headers.Authorization = `Bearer ${this.token}`;
    }

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        if (response.status === 401) {
          this.removeToken();
          window.location.href = '/login';
        }
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Auth methods
  async login(email, password) {
    const data = await this.request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    this.setToken(data.token);
    return data;
  }

  async register(userData) {
    const data = await this.request('/auth/register', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
    this.setToken(data.token);
    return data;
  }

  async logout() {
    this.removeToken();
  }

  // User methods
  async getUsers(params = {}) {
    const queryString = new URLSearchParams(params).toString();
    return this.request(`/users?${queryString}`);
  }

  async getUser(id) {
    return this.request(`/users/${id}`);
  }

  async updateUser(id, userData) {
    return this.request(`/users/${id}`, {
      method: 'PUT',
      body: JSON.stringify(userData),
    });
  }

  async deleteUser(id) {
    return this.request(`/users/${id}`, {
      method: 'DELETE',
    });
  }
}

export default new ApiService();
```

### 2. Real-time Communication

#### WebSocket Implementation
```javascript
// services/websocket.js
class WebSocketService {
  constructor() {
    this.socket = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectInterval = 1000;
    this.listeners = new Map();
  }

  connect(url) {
    return new Promise((resolve, reject) => {
      try {
        this.socket = new WebSocket(url);
        
        this.socket.onopen = () => {
          console.log('WebSocket connected');
          this.reconnectAttempts = 0;
          resolve();
        };

        this.socket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        this.socket.onclose = (event) => {
          console.log('WebSocket disconnected:', event.code, event.reason);
          this.handleReconnect();
        };

        this.socket.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  handleMessage(data) {
    const { type, payload } = data;
    const listeners = this.listeners.get(type) || [];
    listeners.forEach(callback => callback(payload));
  }

  subscribe(eventType, callback) {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, []);
    }
    this.listeners.get(eventType).push(callback);

    // Return unsubscribe function
    return () => {
      const listeners = this.listeners.get(eventType);
      if (listeners) {
        const index = listeners.indexOf(callback);
        if (index > -1) {
          listeners.splice(index, 1);
        }
      }
    };
  }

  send(type, payload) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type, payload }));
    } else {
      console.error('WebSocket is not connected');
    }
  }

  handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      setTimeout(() => {
        this.connect(this.socket.url);
      }, this.reconnectInterval * this.reconnectAttempts);
    } else {
      console.error('Max reconnection attempts reached');
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }
}

export default new WebSocketService();
```

## Web Security

### 1. Authentication & Authorization

#### JWT Implementation
```javascript
// utils/auth.js
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';
const JWT_EXPIRES_IN = '24h';

export const generateToken = (payload) => {
  return jwt.sign(payload, JWT_SECRET, { expiresIn: JWT_EXPIRES_IN });
};

export const verifyToken = (token) => {
  try {
    return jwt.verify(token, JWT_SECRET);
  } catch (error) {
    throw new Error('Invalid token');
  }
};

export const hashPassword = async (password) => {
  const saltRounds = 12;
  return await bcrypt.hash(password, saltRounds);
};

export const comparePassword = async (password, hashedPassword) => {
  return await bcrypt.compare(password, hashedPassword);
};

// Middleware for protecting routes
export const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }

  try {
    const decoded = verifyToken(token);
    req.user = decoded;
    next();
  } catch (error) {
    return res.status(403).json({ error: 'Invalid or expired token' });
  }
};

// Role-based authorization
export const authorize = (roles) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    if (!roles.includes(req.user.role)) {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }

    next();
  };
};
```

### 2. Input Validation & Sanitization

#### Validation Middleware
```javascript
// middleware/validation.js
import { body, param, query, validationResult } from 'express-validator';
import DOMPurify from 'isomorphic-dompurify';

export const validateRequest = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({
      error: 'Validation failed',
      details: errors.array()
    });
  }
  next();
};

export const sanitizeInput = (req, res, next) => {
  // Sanitize string inputs
  const sanitizeObject = (obj) => {
    for (const key in obj) {
      if (typeof obj[key] === 'string') {
        obj[key] = DOMPurify.sanitize(obj[key]);
      } else if (typeof obj[key] === 'object' && obj[key] !== null) {
        sanitizeObject(obj[key]);
      }
    }
  };

  if (req.body) sanitizeObject(req.body);
  if (req.query) sanitizeObject(req.query);
  if (req.params) sanitizeObject(req.params);

  next();
};

// Validation rules
export const userValidationRules = () => {
  return [
    body('email')
      .isEmail()
      .normalizeEmail()
      .withMessage('Valid email is required'),
    body('password')
      .isLength({ min: 8 })
      .matches(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/)
      .withMessage('Password must be at least 8 characters with uppercase, lowercase, number and special character'),
    body('name')
      .trim()
      .isLength({ min: 2, max: 50 })
      .matches(/^[a-zA-Z\s]+$/)
      .withMessage('Name must be 2-50 characters and contain only letters and spaces'),
  ];
};

export const productValidationRules = () => {
  return [
    body('name')
      .trim()
      .isLength({ min: 2, max: 100 })
      .withMessage('Product name must be 2-100 characters'),
    body('price')
      .isFloat({ min: 0 })
      .withMessage('Price must be a positive number'),
    body('description')
      .trim()
      .isLength({ max: 1000 })
      .withMessage('Description must not exceed 1000 characters'),
    body('category')
      .isIn(['electronics', 'clothing', 'books', 'home', 'sports'])
      .withMessage('Invalid category'),
  ];
};
```

## Performance Optimization

### 1. Frontend Optimization

#### Code Splitting & Lazy Loading
```javascript
// App.js
import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

// Lazy load components
const Home = lazy(() => import('./pages/Home'));
const Products = lazy(() => import('./pages/Products'));
const ProductDetail = lazy(() => import('./pages/ProductDetail'));
const Cart = lazy(() => import('./pages/Cart'));
const Profile = lazy(() => import('./pages/Profile'));

// Loading component
const Loading = () => (
  <div className="loading-container">
    <div className="spinner"></div>
    <p>Loading...</p>
  </div>
);

function App() {
  return (
    <Router>
      <div className="App">
        <Suspense fallback={<Loading />}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/products" element={<Products />} />
            <Route path="/products/:id" element={<ProductDetail />} />
            <Route path="/cart" element={<Cart />} />
            <Route path="/profile" element={<Profile />} />
          </Routes>
        </Suspense>
      </div>
    </Router>
  );
}

export default App;
```

#### Image Optimization
```javascript
// components/OptimizedImage.jsx
import React, { useState, useRef, useEffect } from 'react';

const OptimizedImage = ({ 
  src, 
  alt, 
  width, 
  height, 
  placeholder = '/placeholder.jpg',
  className = '',
  ...props 
}) => {
  const [imageSrc, setImageSrc] = useState(placeholder);
  const [isLoaded, setIsLoaded] = useState(false);
  const [isInView, setIsInView] = useState(false);
  const imgRef = useRef();

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1 }
    );

    if (imgRef.current) {
      observer.observe(imgRef.current);
    }

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (isInView) {
      const img = new Image();
      img.onload = () => {
        setImageSrc(src);
        setIsLoaded(true);
      };
      img.src = src;
    }
  }, [isInView, src]);

  return (
    <div 
      ref={imgRef}
      className={`image-container ${className}`}
      style={{ width, height }}
      {...props}
    >
      <img
        src={imageSrc}
        alt={alt}
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          opacity: isLoaded ? 1 : 0.5,
          transition: 'opacity 0.3s ease'
        }}
      />
      {!isLoaded && (
        <div className="image-placeholder">
          <div className="spinner"></div>
        </div>
      )}
    </div>
  );
};

export default OptimizedImage;
```

### 2. Backend Optimization

#### Caching Strategy
```javascript
// middleware/cache.js
import NodeCache from 'node-cache';

const cache = new NodeCache({ 
  stdTTL: 600, // 10 minutes default
  checkperiod: 120 // Check for expired keys every 2 minutes
});

export const cacheMiddleware = (ttl = 600) => {
  return (req, res, next) => {
    const key = req.originalUrl;
    const cachedData = cache.get(key);

    if (cachedData) {
      console.log('Cache hit for:', key);
      return res.json(cachedData);
    }

    // Store original json method
    const originalJson = res.json;

    // Override json method to cache response
    res.json = function(data) {
      cache.set(key, data, ttl);
      console.log('Cached response for:', key);
      return originalJson.call(this, data);
    };

    next();
  };
};

// Redis cache implementation
import Redis from 'ioredis';

const redis = new Redis({
  host: process.env.REDIS_HOST || 'localhost',
  port: process.env.REDIS_PORT || 6379,
  retryDelayOnFailover: 100,
  maxRetriesPerRequest: 3
});

export const redisCache = {
  async get(key) {
    try {
      const data = await redis.get(key);
      return data ? JSON.parse(data) : null;
    } catch (error) {
      console.error('Redis get error:', error);
      return null;
    }
  },

  async set(key, value, ttl = 600) {
    try {
      await redis.setex(key, ttl, JSON.stringify(value));
    } catch (error) {
      console.error('Redis set error:', error);
    }
  },

  async del(key) {
    try {
      await redis.del(key);
    } catch (error) {
      console.error('Redis delete error:', error);
    }
  }
};
```

## Follow-up Questions

### 1. Frontend Development
**Q: What's the difference between controlled and uncontrolled components in React?**
A: Controlled components have their state managed by React (value prop + onChange), while uncontrolled components manage their own state internally (using refs).

### 2. Backend Development
**Q: What are the benefits of using middleware in Express.js?**
A: Middleware provides modular functionality, request/response processing, authentication, logging, error handling, and code reusability across routes.

### 3. Full-Stack Integration
**Q: How do you handle CORS issues in full-stack applications?**
A: Configure CORS middleware on the backend to allow specific origins, methods, and headers, and ensure proper preflight request handling.

## Sources

### Books
- **Full Stack React** by Anthony Accomazzo
- **Node.js in Action** by Mike Cantelon
- **Web Development with Node and Express** by Ethan Brown

### Online Resources
- **React Documentation** - Official React guides
- **Vue.js Guide** - Official Vue.js documentation
- **Express.js Guide** - Express.js best practices
- **MDN Web Docs** - Web standards and APIs

## Projects

### 1. E-commerce Platform
**Objective**: Build a full-stack e-commerce application
**Requirements**: React frontend, Node.js/Go backend, payment integration
**Deliverables**: Complete e-commerce platform with admin panel

### 2. Real-time Chat Application
**Objective**: Create a real-time messaging application
**Requirements**: WebSocket communication, user authentication, message history
**Deliverables**: Chat app with rooms, file sharing, and notifications

### 3. Social Media Dashboard
**Objective**: Build a social media management dashboard
**Requirements**: Multiple API integrations, data visualization, scheduling
**Deliverables**: Dashboard with analytics and content management

---

**Next**: [API Design](./api-design/README.md) | **Previous**: [Database Systems](./database-systems/README.md) | **Up**: [Phase 1](../README.md)

