# 🔄 REST vs GraphQL: API Design Patterns and Trade-offs

> **Master both REST and GraphQL architectures for modern backend development**

## 📚 Concept

### REST (Representational State Transfer)

REST is an architectural style that uses HTTP methods to perform operations on resources. It's stateless, cacheable, and follows a uniform interface.

**Key Principles:**

- **Stateless**: Each request contains all information needed
- **Client-Server**: Separation of concerns
- **Cacheable**: Responses can be cached
- **Uniform Interface**: Consistent API design
- **Layered System**: Hierarchical layers

### GraphQL

GraphQL is a query language and runtime for APIs that allows clients to request exactly the data they need.

**Key Features:**

- **Single Endpoint**: One endpoint for all operations
- **Strongly Typed**: Schema defines data structure
- **Client-Driven**: Clients specify required fields
- **Real-time**: Subscriptions for live updates
- **Introspection**: Self-documenting API

## 🏗️ Architecture Comparison

### REST Architecture

```
Client ──HTTP──► API Gateway ──► Microservice A ──► Database A
                │
                ├──► Microservice B ──► Database B
                │
                └──► Microservice C ──► Database C
```

### GraphQL Architecture

```
Client ──GraphQL──► GraphQL Gateway ──► Resolvers ──► Multiple Data Sources
                                    │
                                    ├──► REST APIs
                                    ├──► Databases
                                    ├──► Microservices
                                    └──► External APIs
```

## 🛠️ Hands-on Example

### REST API Implementation (Go)

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "strconv"
    "strings"
    "time"
)

type User struct {
    ID       int       `json:"id"`
    Name     string    `json:"name"`
    Email    string    `json:"email"`
    Posts    []Post    `json:"posts,omitempty"`
    CreatedAt time.Time `json:"created_at"`
}

type Post struct {
    ID        int       `json:"id"`
    Title     string    `json:"title"`
    Content   string    `json:"content"`
    UserID    int       `json:"user_id"`
    CreatedAt time.Time `json:"created_at"`
}

type UserService struct {
    users map[int]User
    posts map[int][]Post
}

func NewUserService() *UserService {
    return &UserService{
        users: map[int]User{
            1: {ID: 1, Name: "John Doe", Email: "john@example.com", CreatedAt: time.Now()},
            2: {ID: 2, Name: "Jane Smith", Email: "jane@example.com", CreatedAt: time.Now()},
        },
        posts: map[int][]Post{
            1: {
                {ID: 1, Title: "My First Post", Content: "Hello World!", UserID: 1, CreatedAt: time.Now()},
                {ID: 2, Title: "Learning Go", Content: "Go is awesome!", UserID: 1, CreatedAt: time.Now()},
            },
            2: {
                {ID: 3, Title: "GraphQL vs REST", Content: "Both have their place", UserID: 2, CreatedAt: time.Now()},
            },
        },
    }
}

// GET /api/users
func (us *UserService) GetUsers(w http.ResponseWriter, r *http.Request) {
    // Query parameters for filtering and pagination
    pageStr := r.URL.Query().Get("page")
    limitStr := r.URL.Query().Get("limit")
    includePosts := r.URL.Query().Get("include_posts") == "true"

    page, _ := strconv.Atoi(pageStr)
    limit, _ := strconv.Atoi(limitStr)

    if page <= 0 {
        page = 1
    }
    if limit <= 0 {
        limit = 10
    }

    // Get all users
    allUsers := make([]User, 0, len(us.users))
    for _, user := range us.users {
        if includePosts {
            user.Posts = us.posts[user.ID]
        }
        allUsers = append(allUsers, user)
    }

    // Simple pagination
    start := (page - 1) * limit
    end := start + limit

    if start >= len(allUsers) {
        allUsers = []User{}
    } else if end > len(allUsers) {
        allUsers = allUsers[start:]
    } else {
        allUsers = allUsers[start:end]
    }

    response := map[string]interface{}{
        "users": allUsers,
        "pagination": map[string]interface{}{
            "page":  page,
            "limit": limit,
            "total": len(us.users),
        },
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

// GET /api/users/{id}
func (us *UserService) GetUser(w http.ResponseWriter, r *http.Request) {
    pathParts := strings.Split(r.URL.Path, "/")
    if len(pathParts) < 4 {
        http.Error(w, "Invalid URL", http.StatusBadRequest)
        return
    }

    id, err := strconv.Atoi(pathParts[3])
    if err != nil {
        http.Error(w, "Invalid user ID", http.StatusBadRequest)
        return
    }

    user, exists := us.users[id]
    if !exists {
        http.Error(w, "User not found", http.StatusNotFound)
        return
    }

    // Check if posts should be included
    includePosts := r.URL.Query().Get("include_posts") == "true"
    if includePosts {
        user.Posts = us.posts[user.ID]
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

// GET /api/users/{id}/posts
func (us *UserService) GetUserPosts(w http.ResponseWriter, r *http.Request) {
    pathParts := strings.Split(r.URL.Path, "/")
    if len(pathParts) < 5 {
        http.Error(w, "Invalid URL", http.StatusBadRequest)
        return
    }

    id, err := strconv.Atoi(pathParts[3])
    if err != nil {
        http.Error(w, "Invalid user ID", http.StatusBadRequest)
        return
    }

    if _, exists := us.users[id]; !exists {
        http.Error(w, "User not found", http.StatusNotFound)
        return
    }

    posts := us.posts[id]

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]interface{}{
        "posts": posts,
        "user_id": id,
    })
}

// POST /api/users
func (us *UserService) CreateUser(w http.ResponseWriter, r *http.Request) {
    var user User
    if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }

    // Generate new ID
    user.ID = len(us.users) + 1
    user.CreatedAt = time.Now()
    us.users[user.ID] = user

    w.Header().Set("Content-Type", "application/json")
    w.Header().Set("Location", fmt.Sprintf("/api/users/%d", user.ID))
    w.WriteHeader(http.StatusCreated)
    json.NewEncoder(w).Encode(user)
}

func main() {
    userService := NewUserService()

    // REST endpoints
    http.HandleFunc("/api/users", func(w http.ResponseWriter, r *http.Request) {
        switch r.Method {
        case http.MethodGet:
            userService.GetUsers(w, r)
        case http.MethodPost:
            userService.CreateUser(w, r)
        default:
            http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        }
    })

    http.HandleFunc("/api/users/", func(w http.ResponseWriter, r *http.Request) {
        if strings.HasSuffix(r.URL.Path, "/posts") {
            userService.GetUserPosts(w, r)
        } else {
            userService.GetUser(w, r)
        }
    })

    log.Println("REST API server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

### GraphQL Implementation (Go with gqlgen)

```go
//go:generate go run github.com/99designs/gqlgen

package main

import (
    "context"
    "fmt"
    "log"
    "net/http"
    "time"

    "github.com/99designs/gqlgen/graphql/handler"
    "github.com/99designs/gqlgen/graphql/playground"
    "github.com/gorilla/mux"
)

// GraphQL Schema
const schema = `
type User {
    id: ID!
    name: String!
    email: String!
    posts: [Post!]!
    createdAt: String!
}

type Post {
    id: ID!
    title: String!
    content: String!
    user: User!
    createdAt: String!
}

type Query {
    users(page: Int, limit: Int): [User!]!
    user(id: ID!): User
    posts(userId: ID): [Post!]!
}

type Mutation {
    createUser(input: CreateUserInput!): User!
    createPost(input: CreatePostInput!): Post!
}

input CreateUserInput {
    name: String!
    email: String!
}

input CreatePostInput {
    title: String!
    content: String!
    userId: ID!
}
`

// GraphQL Resolvers
type Resolver struct {
    users map[int]User
    posts map[int][]Post
}

func NewResolver() *Resolver {
    return &Resolver{
        users: map[int]User{
            1: {ID: 1, Name: "John Doe", Email: "john@example.com", CreatedAt: time.Now()},
            2: {ID: 2, Name: "Jane Smith", Email: "jane@example.com", CreatedAt: time.Now()},
        },
        posts: map[int][]Post{
            1: {
                {ID: 1, Title: "My First Post", Content: "Hello World!", UserID: 1, CreatedAt: time.Now()},
                {ID: 2, Title: "Learning Go", Content: "Go is awesome!", UserID: 1, CreatedAt: time.Now()},
            },
            2: {
                {ID: 3, Title: "GraphQL vs REST", Content: "Both have their place", UserID: 2, CreatedAt: time.Now()},
            },
        },
    }
}

// Query resolvers
func (r *Resolver) Users(ctx context.Context, page *int, limit *int) ([]User, error) {
    allUsers := make([]User, 0, len(r.users))
    for _, user := range r.users {
        user.Posts = r.posts[user.ID]
        allUsers = append(allUsers, user)
    }

    // Apply pagination if specified
    if page != nil && limit != nil {
        start := (*page - 1) * *limit
        end := start + *limit

        if start >= len(allUsers) {
            return []User{}, nil
        }
        if end > len(allUsers) {
            return allUsers[start:], nil
        }
        return allUsers[start:end], nil
    }

    return allUsers, nil
}

func (r *Resolver) User(ctx context.Context, id string) (*User, error) {
    userID, err := strconv.Atoi(id)
    if err != nil {
        return nil, fmt.Errorf("invalid user ID")
    }

    user, exists := r.users[userID]
    if !exists {
        return nil, fmt.Errorf("user not found")
    }

    user.Posts = r.posts[user.ID]
    return &user, nil
}

func (r *Resolver) Posts(ctx context.Context, userID *string) ([]Post, error) {
    if userID != nil {
        id, err := strconv.Atoi(*userID)
        if err != nil {
            return nil, fmt.Errorf("invalid user ID")
        }
        return r.posts[id], nil
    }

    // Return all posts
    allPosts := make([]Post, 0)
    for _, posts := range r.posts {
        allPosts = append(allPosts, posts...)
    }
    return allPosts, nil
}

// Mutation resolvers
func (r *Resolver) CreateUser(ctx context.Context, input CreateUserInput) (User, error) {
    user := User{
        ID:        len(r.users) + 1,
        Name:      input.Name,
        Email:     input.Email,
        CreatedAt: time.Now(),
    }

    r.users[user.ID] = user
    return user, nil
}

func (r *Resolver) CreatePost(ctx context.Context, input CreatePostInput) (Post, error) {
    userID, err := strconv.Atoi(input.UserID)
    if err != nil {
        return Post{}, fmt.Errorf("invalid user ID")
    }

    if _, exists := r.users[userID]; !exists {
        return Post{}, fmt.Errorf("user not found")
    }

    post := Post{
        ID:        len(r.posts[userID]) + 1,
        Title:     input.Title,
        Content:   input.Content,
        UserID:    userID,
        CreatedAt: time.Now(),
    }

    r.posts[userID] = append(r.posts[userID], post)
    return post, nil
}

func main() {
    resolver := NewResolver()

    // GraphQL handler
    srv := handler.NewDefaultServer(NewExecutableSchema(Config{Resolvers: resolver}))

    router := mux.NewRouter()
    router.Handle("/", playground.Handler("GraphQL playground", "/query"))
    router.Handle("/query", srv)

    log.Println("GraphQL server starting on :8081")
    log.Fatal(http.ListenAndServe(":8081", router))
}
```

## 📊 Performance Comparison

### REST API Calls

```bash
# Multiple requests needed for related data
curl http://localhost:8080/api/users/1
curl http://localhost:8080/api/users/1/posts
curl http://localhost:8080/api/users/2
curl http://localhost:8080/api/users/2/posts

# Total: 4 HTTP requests, potential over-fetching
```

### GraphQL Query

```graphql
# Single request for exactly what's needed
query GetUsersWithPosts {
  users {
    id
    name
    email
    posts {
      id
      title
    }
  }
}
```

## 🔍 Query Examples

### REST API Usage

```bash
# Get all users with pagination
curl "http://localhost:8080/api/users?page=1&limit=10&include_posts=true"

# Get specific user
curl "http://localhost:8080/api/users/1?include_posts=true"

# Get user's posts
curl "http://localhost:8080/api/users/1/posts"

# Create user
curl -X POST http://localhost:8080/api/users \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice Johnson", "email": "alice@example.com"}'
```

### GraphQL Queries

```graphql
# Get users with posts
query GetUsersWithPosts {
  users {
    id
    name
    email
    posts {
      id
      title
    }
  }
}

# Get specific user with posts
query GetUser($id: ID!) {
  user(id: $id) {
    id
    name
    email
    posts {
      id
      title
      content
    }
  }
}

# Get posts with user information
query GetPostsWithUsers {
  posts {
    id
    title
    content
    user {
      id
      name
    }
  }
}

# Create user mutation
mutation CreateUser($input: CreateUserInput!) {
  createUser(input: $input) {
    id
    name
    email
    createdAt
  }
}

# Create post mutation
mutation CreatePost($input: CreatePostInput!) {
  createPost(input: $input) {
    id
    title
    content
    user {
      id
      name
    }
  }
}
```

## ⚖️ Trade-offs Analysis

### REST Advantages

- **Simplicity**: Easy to understand and implement
- **Caching**: HTTP caching works out of the box
- **Stateless**: Each request is independent
- **Mature**: Well-established patterns and tools
- **HTTP Standards**: Leverages existing HTTP infrastructure

### REST Disadvantages

- **Over-fetching**: Get more data than needed
- **Under-fetching**: Multiple requests for related data
- **Versioning**: API versioning can be complex
- **Multiple Endpoints**: Different URLs for different resources

### GraphQL Advantages

- **Flexible Queries**: Request exactly what you need
- **Single Endpoint**: One endpoint for all operations
- **Strong Typing**: Schema provides type safety
- **Real-time**: Built-in subscription support
- **Introspection**: Self-documenting API

### GraphQL Disadvantages

- **Complexity**: Steeper learning curve
- **Caching**: More complex caching strategies
- **N+1 Problem**: Potential performance issues
- **File Uploads**: Not natively supported
- **Learning Curve**: Team needs to learn GraphQL

## 🚀 Best Practices

### REST Best Practices

```go
// 1. Use proper HTTP methods
func (s *Service) HandleUser(w http.ResponseWriter, r *http.Request) {
    switch r.Method {
    case http.MethodGet:
        s.getUser(w, r)
    case http.MethodPost:
        s.createUser(w, r)
    case http.MethodPut:
        s.updateUser(w, r)
    case http.MethodDelete:
        s.deleteUser(w, r)
    default:
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
    }
}

// 2. Implement proper error handling
func (s *Service) getUser(w http.ResponseWriter, r *http.Request) {
    user, err := s.userRepo.GetByID(userID)
    if err != nil {
        if errors.Is(err, ErrUserNotFound) {
            http.Error(w, "User not found", http.StatusNotFound)
        } else {
            http.Error(w, "Internal server error", http.StatusInternalServerError)
        }
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

// 3. Add proper headers
func (s *Service) createUser(w http.ResponseWriter, r *http.Request) {
    user, err := s.userRepo.Create(userData)
    if err != nil {
        http.Error(w, "Failed to create user", http.StatusBadRequest)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    w.Header().Set("Location", fmt.Sprintf("/api/users/%d", user.ID))
    w.WriteHeader(http.StatusCreated)
    json.NewEncoder(w).Encode(user)
}
```

### GraphQL Best Practices

```go
// 1. Implement DataLoader for N+1 problem
type UserLoader struct {
    userRepo UserRepository
    cache    map[int]User
    mutex    sync.RWMutex
}

func (ul *UserLoader) Load(ctx context.Context, userID int) (User, error) {
    ul.mutex.RLock()
    if user, exists := ul.cache[userID]; exists {
        ul.mutex.RUnlock()
        return user, nil
    }
    ul.mutex.RUnlock()

    user, err := ul.userRepo.GetByID(userID)
    if err != nil {
        return User{}, err
    }

    ul.mutex.Lock()
    ul.cache[userID] = user
    ul.mutex.Unlock()

    return user, nil
}

// 2. Implement proper error handling
func (r *Resolver) User(ctx context.Context, id string) (*User, error) {
    userID, err := strconv.Atoi(id)
    if err != nil {
        return nil, &gqlerror.Error{
            Message: "Invalid user ID",
            Extensions: map[string]interface{}{
                "code": "INVALID_USER_ID",
            },
        }
    }

    user, err := r.userLoader.Load(ctx, userID)
    if err != nil {
        if errors.Is(err, ErrUserNotFound) {
            return nil, &gqlerror.Error{
                Message: "User not found",
                Extensions: map[string]interface{}{
                    "code": "USER_NOT_FOUND",
                },
            }
        }
        return nil, err
    }

    return &user, nil
}

// 3. Implement query complexity analysis
func (r *Resolver) Users(ctx context.Context, page *int, limit *int) ([]User, error) {
    // Check query complexity
    if limit != nil && *limit > 100 {
        return nil, &gqlerror.Error{
            Message: "Limit too high",
            Extensions: map[string]interface{}{
                "code": "LIMIT_TOO_HIGH",
            },
        }
    }

    return r.userRepo.GetUsers(page, limit)
}
```

## 🏢 Industry Insights

### When to Use REST

- **Public APIs**: External-facing APIs for third-party developers
- **Simple CRUD**: Basic create, read, update, delete operations
- **Caching Critical**: When HTTP caching is important
- **Team Familiarity**: Team is more familiar with REST
- **Microservices**: Service-to-service communication

### When to Use GraphQL

- **Mobile Apps**: Reduce data usage with flexible queries
- **Complex Data Relationships**: Multiple related entities
- **Real-time Features**: Need for subscriptions
- **Rapid Frontend Development**: Frequent UI changes
- **Data Aggregation**: Combining multiple data sources

### Company Examples

- **Facebook**: GraphQL for mobile apps and complex data relationships
- **GitHub**: GraphQL API for flexible data fetching
- **Shopify**: GraphQL for e-commerce APIs
- **Netflix**: REST for external APIs, GraphQL for internal tools
- **Twitter**: REST for public API, GraphQL for internal services

## 🎯 Interview Questions

### Basic Level

1. **What's the difference between REST and GraphQL?**

   - REST uses multiple endpoints, GraphQL uses single endpoint
   - REST follows HTTP methods, GraphQL uses queries/mutations
   - REST can over-fetch/under-fetch, GraphQL requests exact data

2. **When would you choose REST over GraphQL?**

   - Simple CRUD operations
   - Public APIs for third-party developers
   - When HTTP caching is critical
   - Team familiarity with REST

3. **What are the main HTTP methods in REST?**
   - GET: Retrieve data
   - POST: Create resource
   - PUT: Update/replace resource
   - DELETE: Remove resource
   - PATCH: Partial update

### Intermediate Level

4. **How do you handle the N+1 problem in GraphQL?**

   ```go
   // Use DataLoader to batch requests
   type PostLoader struct {
       postRepo PostRepository
       cache    map[int][]Post
   }

   func (pl *PostLoader) Load(ctx context.Context, userID int) ([]Post, error) {
       if posts, exists := pl.cache[userID]; exists {
           return posts, nil
       }

       posts, err := pl.postRepo.GetByUserID(userID)
       if err != nil {
           return nil, err
       }

       pl.cache[userID] = posts
       return posts, nil
   }
   ```

5. **How do you implement caching in GraphQL?**

   - **Query-level caching**: Cache entire query results
   - **Field-level caching**: Cache individual field resolvers
   - **DataLoader caching**: Cache database queries
   - **HTTP caching**: Use CDN for static queries

6. **Explain GraphQL subscriptions?**
   ```graphql
   subscription OnPostCreated {
     postCreated {
       id
       title
       user {
         name
       }
     }
   }
   ```

### Advanced Level

7. **How do you implement rate limiting in GraphQL?**

   ```go
   func RateLimitMiddleware(next http.Handler) http.Handler {
       return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
           // Parse GraphQL query
           var req struct {
               Query string `json:"query"`
           }
           json.NewDecoder(r.Body).Decode(&req)

           // Calculate query complexity
           complexity := calculateComplexity(req.Query)

           // Check rate limit based on complexity
           if !rateLimiter.Allow(complexity) {
               http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
               return
           }

           next.ServeHTTP(w, r)
       })
   }
   ```

8. **How do you handle file uploads in GraphQL?**

   - Use multipart/form-data for file uploads
   - Implement custom scalar type for file handling
   - Use base64 encoding for small files
   - Use signed URLs for large files

9. **Design a hybrid REST/GraphQL architecture?**
   - Use REST for external APIs and file uploads
   - Use GraphQL for internal services and complex queries
   - Implement API gateway to route requests
   - Use GraphQL to aggregate REST APIs

---

**Next**: [Authentication](./Authentication.md) - JWT, OAuth2, and session management
