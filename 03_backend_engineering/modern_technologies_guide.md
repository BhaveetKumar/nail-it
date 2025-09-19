# 🚀 Modern Technologies Guide

> **Essential modern backend technologies: GraphQL, gRPC, Event Streaming, and more**

## 🎯 **Overview**

This guide covers cutting-edge technologies that are increasingly important in modern backend engineering interviews, especially for senior and staff-level positions at companies like Razorpay, Meta, Google, and other tech giants.

## 📚 **Table of Contents**

1. [GraphQL](#graphql)
2. [gRPC](#grpc)
3. [Event Streaming](#event-streaming)
4. [Message Brokers](#message-brokers)
5. [Real-time Technologies](#real-time-technologies)
6. [Modern Database Technologies](#modern-database-technologies)
7. [Interview Questions](#interview-questions)

---

## 🔍 **GraphQL**

### **What is GraphQL?**

GraphQL is a query language and runtime for APIs that allows clients to request exactly the data they need, nothing more, nothing less.

### **Core Concepts**

```graphql
# Schema Definition Language (SDL)
type User {
  id: ID!
  name: String!
  email: String!
  posts: [Post!]!
  createdAt: DateTime!
}

type Post {
  id: ID!
  title: String!
  content: String!
  author: User!
  comments: [Comment!]!
  publishedAt: DateTime
}

type Comment {
  id: ID!
  content: String!
  author: User!
  post: Post!
  createdAt: DateTime!
}

# Root Types
type Query {
  user(id: ID!): User
  users(limit: Int, offset: Int): [User!]!
  post(id: ID!): Post
  posts(authorId: ID, published: Boolean): [Post!]!
}

type Mutation {
  createUser(input: CreateUserInput!): User!
  updateUser(id: ID!, input: UpdateUserInput!): User!
  deleteUser(id: ID!): Boolean!
  
  createPost(input: CreatePostInput!): Post!
  publishPost(id: ID!): Post!
}

type Subscription {
  postAdded(authorId: ID): Post!
  commentAdded(postId: ID!): Comment!
  userOnline: User!
}

# Input Types
input CreateUserInput {
  name: String!
  email: String!
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
```

### **GraphQL Server Implementation (Go)**

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "time"

    "github.com/graphql-go/graphql"
    "github.com/graphql-go/handler"
)

// Models
type User struct {
    ID        string    `json:"id"`
    Name      string    `json:"name"`
    Email     string    `json:"email"`
    CreatedAt time.Time `json:"createdAt"`
}

type Post struct {
    ID          string    `json:"id"`
    Title       string    `json:"title"`
    Content     string    `json:"content"`
    AuthorID    string    `json:"authorId"`
    PublishedAt *time.Time `json:"publishedAt"`
    CreatedAt   time.Time `json:"createdAt"`
}

// Data Layer
type DataLayer struct {
    users map[string]*User
    posts map[string]*Post
}

func NewDataLayer() *DataLayer {
    return &DataLayer{
        users: make(map[string]*User),
        posts: make(map[string]*Post),
    }
}

func (dl *DataLayer) GetUser(id string) *User {
    return dl.users[id]
}

func (dl *DataLayer) GetUsers() []*User {
    users := make([]*User, 0, len(dl.users))
    for _, user := range dl.users {
        users = append(users, user)
    }
    return users
}

func (dl *DataLayer) GetPostsByAuthor(authorID string) []*Post {
    posts := make([]*Post, 0)
    for _, post := range dl.posts {
        if post.AuthorID == authorID {
            posts = append(posts, post)
        }
    }
    return posts
}

func (dl *DataLayer) CreateUser(name, email string) *User {
    user := &User{
        ID:        fmt.Sprintf("user_%d", len(dl.users)+1),
        Name:      name,
        Email:     email,
        CreatedAt: time.Now(),
    }
    dl.users[user.ID] = user
    return user
}

// GraphQL Schema
func createSchema(dataLayer *DataLayer) (graphql.Schema, error) {
    // User Type
    userType := graphql.NewObject(graphql.ObjectConfig{
        Name: "User",
        Fields: graphql.Fields{
            "id": &graphql.Field{
                Type: graphql.NewNonNull(graphql.String),
            },
            "name": &graphql.Field{
                Type: graphql.NewNonNull(graphql.String),
            },
            "email": &graphql.Field{
                Type: graphql.NewNonNull(graphql.String),
            },
            "createdAt": &graphql.Field{
                Type: graphql.DateTime,
            },
            "posts": &graphql.Field{
                Type: graphql.NewList(postType),
                Resolve: func(p graphql.ResolveParams) (interface{}, error) {
                    user := p.Source.(*User)
                    return dataLayer.GetPostsByAuthor(user.ID), nil
                },
            },
        },
    })

    // Post Type
    postType := graphql.NewObject(graphql.ObjectConfig{
        Name: "Post",
        Fields: graphql.Fields{
            "id": &graphql.Field{
                Type: graphql.NewNonNull(graphql.String),
            },
            "title": &graphql.Field{
                Type: graphql.NewNonNull(graphql.String),
            },
            "content": &graphql.Field{
                Type: graphql.NewNonNull(graphql.String),
            },
            "publishedAt": &graphql.Field{
                Type: graphql.DateTime,
            },
            "author": &graphql.Field{
                Type: userType,
                Resolve: func(p graphql.ResolveParams) (interface{}, error) {
                    post := p.Source.(*Post)
                    return dataLayer.GetUser(post.AuthorID), nil
                },
            },
        },
    })

    // Root Query
    rootQuery := graphql.NewObject(graphql.ObjectConfig{
        Name: "Query",
        Fields: graphql.Fields{
            "user": &graphql.Field{
                Type: userType,
                Args: graphql.FieldConfigArgument{
                    "id": &graphql.ArgumentConfig{
                        Type: graphql.NewNonNull(graphql.String),
                    },
                },
                Resolve: func(p graphql.ResolveParams) (interface{}, error) {
                    id := p.Args["id"].(string)
                    return dataLayer.GetUser(id), nil
                },
            },
            "users": &graphql.Field{
                Type: graphql.NewList(userType),
                Args: graphql.FieldConfigArgument{
                    "limit": &graphql.ArgumentConfig{
                        Type: graphql.Int,
                    },
                    "offset": &graphql.ArgumentConfig{
                        Type: graphql.Int,
                    },
                },
                Resolve: func(p graphql.ResolveParams) (interface{}, error) {
                    users := dataLayer.GetUsers()
                    
                    // Apply pagination
                    offset := 0
                    if val, ok := p.Args["offset"]; ok {
                        offset = val.(int)
                    }
                    
                    limit := len(users)
                    if val, ok := p.Args["limit"]; ok {
                        limit = val.(int)
                    }
                    
                    end := offset + limit
                    if end > len(users) {
                        end = len(users)
                    }
                    
                    if offset >= len(users) {
                        return []*User{}, nil
                    }
                    
                    return users[offset:end], nil
                },
            },
        },
    })

    // Root Mutation
    rootMutation := graphql.NewObject(graphql.ObjectConfig{
        Name: "Mutation",
        Fields: graphql.Fields{
            "createUser": &graphql.Field{
                Type: userType,
                Args: graphql.FieldConfigArgument{
                    "name": &graphql.ArgumentConfig{
                        Type: graphql.NewNonNull(graphql.String),
                    },
                    "email": &graphql.ArgumentConfig{
                        Type: graphql.NewNonNull(graphql.String),
                    },
                },
                Resolve: func(p graphql.ResolveParams) (interface{}, error) {
                    name := p.Args["name"].(string)
                    email := p.Args["email"].(string)
                    return dataLayer.CreateUser(name, email), nil
                },
            },
        },
    })

    return graphql.NewSchema(graphql.SchemaConfig{
        Query:    rootQuery,
        Mutation: rootMutation,
    })
}

// GraphQL Middleware
func corsMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
        
        if r.Method == "OPTIONS" {
            w.WriteHeader(http.StatusOK)
            return
        }
        
        next.ServeHTTP(w, r)
    })
}

// DataLoader Pattern for N+1 Problem
type UserLoader struct {
    dataLayer *DataLayer
    cache     map[string]*User
}

func NewUserLoader(dataLayer *DataLayer) *UserLoader {
    return &UserLoader{
        dataLayer: dataLayer,
        cache:     make(map[string]*User),
    }
}

func (ul *UserLoader) Load(ctx context.Context, userID string) (*User, error) {
    if user, exists := ul.cache[userID]; exists {
        return user, nil
    }
    
    user := ul.dataLayer.GetUser(userID)
    if user != nil {
        ul.cache[userID] = user
    }
    
    return user, nil
}

func main() {
    dataLayer := NewDataLayer()
    
    // Seed data
    user1 := dataLayer.CreateUser("John Doe", "john@example.com")
    user2 := dataLayer.CreateUser("Jane Smith", "jane@example.com")
    
    schema, err := createSchema(dataLayer)
    if err != nil {
        log.Fatal(err)
    }

    h := handler.New(&handler.Config{
        Schema:   &schema,
        Pretty:   true,
        GraphiQL: true,
    })

    http.Handle("/graphql", corsMiddleware(h))
    
    fmt.Println("GraphQL server running on :8080/graphql")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

### **GraphQL Client Implementation (Node.js)**

```javascript
// GraphQL Client with Apollo
const { ApolloClient, InMemoryCache, gql, createHttpLink } = require('@apollo/client');
const fetch = require('cross-fetch');

const httpLink = createHttpLink({
    uri: 'http://localhost:8080/graphql',
    fetch: fetch
});

const client = new ApolloClient({
    link: httpLink,
    cache: new InMemoryCache(),
    defaultOptions: {
        watchQuery: {
            errorPolicy: 'all'
        },
        query: {
            errorPolicy: 'all'
        }
    }
});

// Queries
const GET_USERS = gql`
    query GetUsers($limit: Int, $offset: Int) {
        users(limit: $limit, offset: $offset) {
            id
            name
            email
            createdAt
            posts {
                id
                title
                publishedAt
            }
        }
    }
`;

const GET_USER = gql`
    query GetUser($id: ID!) {
        user(id: $id) {
            id
            name
            email
            posts {
                id
                title
                content
                publishedAt
            }
        }
    }
`;

// Mutations
const CREATE_USER = gql`
    mutation CreateUser($name: String!, $email: String!) {
        createUser(name: $name, email: $email) {
            id
            name
            email
            createdAt
        }
    }
`;

// Usage Examples
async function fetchUsers() {
    try {
        const { data, errors } = await client.query({
            query: GET_USERS,
            variables: { limit: 10, offset: 0 }
        });
        
        if (errors) {
            console.error('GraphQL errors:', errors);
        }
        
        console.log('Users:', data.users);
        return data.users;
    } catch (error) {
        console.error('Network error:', error);
        throw error;
    }
}

async function createUser(name, email) {
    try {
        const { data } = await client.mutate({
            mutation: CREATE_USER,
            variables: { name, email },
            update: (cache, { data: { createUser } }) => {
                // Update cache
                const existingUsers = cache.readQuery({ 
                    query: GET_USERS,
                    variables: { limit: 10, offset: 0 }
                });
                
                if (existingUsers) {
                    cache.writeQuery({
                        query: GET_USERS,
                        variables: { limit: 10, offset: 0 },
                        data: {
                            users: [...existingUsers.users, createUser]
                        }
                    });
                }
            }
        });
        
        console.log('Created user:', data.createUser);
        return data.createUser;
    } catch (error) {
        console.error('Create user error:', error);
        throw error;
    }
}

// Subscriptions (WebSocket)
const { WebSocketLink } = require('@apollo/client/link/ws');
const { split } = require('@apollo/client');
const { getMainDefinition } = require('@apollo/client/utilities');

const wsLink = new WebSocketLink({
    uri: 'ws://localhost:8080/graphql',
    options: {
        reconnect: true
    }
});

const splitLink = split(
    ({ query }) => {
        const definition = getMainDefinition(query);
        return (
            definition.kind === 'OperationDefinition' &&
            definition.operation === 'subscription'
        );
    },
    wsLink,
    httpLink
);

const clientWithSubscriptions = new ApolloClient({
    link: splitLink,
    cache: new InMemoryCache()
});

// Subscription example
const POST_ADDED = gql`
    subscription PostAdded($authorId: ID) {
        postAdded(authorId: $authorId) {
            id
            title
            content
            author {
                id
                name
            }
        }
    }
`;

const subscription = clientWithSubscriptions.subscribe({
    query: POST_ADDED,
    variables: { authorId: "user_1" }
}).subscribe({
    next: ({ data }) => {
        console.log('New post:', data.postAdded);
    },
    error: (error) => {
        console.error('Subscription error:', error);
    }
});
```

### **GraphQL Best Practices**

```javascript
// 1. Query Complexity Analysis
const depthLimit = require('graphql-depth-limit');
const costAnalysis = require('graphql-cost-analysis');

const server = new ApolloServer({
    typeDefs,
    resolvers,
    validationRules: [
        depthLimit(7), // Limit query depth
        costAnalysis({
            maximumCost: 1000,
            createError: (max, actual) => {
                return new Error(`Query cost ${actual} exceeds maximum cost ${max}`);
            }
        })
    ]
});

// 2. Caching Strategies
const DataLoader = require('dataloader');

class UserService {
    constructor() {
        this.userLoader = new DataLoader(this.batchGetUsers.bind(this));
    }
    
    async batchGetUsers(userIds) {
        // Batch load users from database
        const users = await User.findByIds(userIds);
        
        // Return users in the same order as requested IDs
        return userIds.map(id => users.find(user => user.id === id));
    }
    
    async getUser(id) {
        return this.userLoader.load(id);
    }
}

// 3. Error Handling
const { UserInputError, AuthenticationError, ForbiddenError } = require('apollo-server');

const resolvers = {
    Query: {
        user: async (parent, { id }, context) => {
            if (!context.user) {
                throw new AuthenticationError('You must be logged in');
            }
            
            if (!id || typeof id !== 'string') {
                throw new UserInputError('Invalid user ID format');
            }
            
            const user = await UserService.getUser(id);
            
            if (!user) {
                throw new UserInputError('User not found');
            }
            
            if (!context.user.canViewUser(user)) {
                throw new ForbiddenError('Insufficient permissions');
            }
            
            return user;
        }
    }
};

// 4. Schema Stitching and Federation
const { buildFederatedSchema } = require('@apollo/federation');

const typeDefs = gql`
    type User @key(fields: "id") {
        id: ID!
        name: String!
        email: String!
    }
    
    extend type Post @key(fields: "id") {
        id: ID! @external
        author: User
    }
`;

const resolvers = {
    User: {
        __resolveReference(user) {
            return UserService.getUser(user.id);
        }
    },
    Post: {
        author(post) {
            return { __typename: "User", id: post.authorId };
        }
    }
};

const schema = buildFederatedSchema([{ typeDefs, resolvers }]);
```

---

## 🔗 **gRPC**

### **What is gRPC?**

gRPC is a high-performance, open-source RPC framework that uses HTTP/2 for transport, Protocol Buffers as the interface description language, and provides features like authentication, bidirectional streaming, and flow control.

### **Protocol Buffer Definition**

```protobuf
// user.proto
syntax = "proto3";

package user.v1;

import "google/protobuf/timestamp.proto";
import "google/protobuf/empty.proto";

option go_package = "github.com/example/user/v1;userv1";

// User service definition
service UserService {
  // Unary RPC
  rpc GetUser(GetUserRequest) returns (GetUserResponse);
  rpc CreateUser(CreateUserRequest) returns (CreateUserResponse);
  rpc UpdateUser(UpdateUserRequest) returns (UpdateUserResponse);
  rpc DeleteUser(DeleteUserRequest) returns (google.protobuf.Empty);
  
  // Server streaming RPC
  rpc ListUsers(ListUsersRequest) returns (stream User);
  
  // Client streaming RPC
  rpc CreateMultipleUsers(stream CreateUserRequest) returns (CreateMultipleUsersResponse);
  
  // Bidirectional streaming RPC
  rpc ChatWithUser(stream ChatMessage) returns (stream ChatMessage);
}

// Messages
message User {
  string id = 1;
  string name = 2;
  string email = 3;
  google.protobuf.Timestamp created_at = 4;
  google.protobuf.Timestamp updated_at = 5;
  repeated string roles = 6;
  map<string, string> metadata = 7;
}

message GetUserRequest {
  string id = 1;
}

message GetUserResponse {
  User user = 1;
}

message CreateUserRequest {
  string name = 1;
  string email = 2;
  repeated string roles = 3;
  map<string, string> metadata = 4;
}

message CreateUserResponse {
  User user = 1;
}

message UpdateUserRequest {
  string id = 1;
  optional string name = 2;
  optional string email = 3;
  repeated string roles = 4;
  map<string, string> metadata = 5;
}

message UpdateUserResponse {
  User user = 1;
}

message DeleteUserRequest {
  string id = 1;
}

message ListUsersRequest {
  int32 page_size = 1;
  string page_token = 2;
  string filter = 3;
  string order_by = 4;
}

message CreateMultipleUsersResponse {
  repeated User users = 1;
  int32 created_count = 2;
}

message ChatMessage {
  string user_id = 1;
  string message = 2;
  google.protobuf.Timestamp timestamp = 3;
}

// Error details
message ErrorDetail {
  string code = 1;
  string message = 2;
  map<string, string> metadata = 3;
}
```

### **gRPC Server Implementation (Go)**

```go
package main

import (
    "context"
    "fmt"
    "io"
    "log"
    "net"
    "sync"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
    "google.golang.org/protobuf/types/known/emptypb"
    "google.golang.org/protobuf/types/known/timestamppb"
    
    userv1 "github.com/example/user/v1" // Generated protobuf code
)

// UserServer implements the UserService
type UserServer struct {
    userv1.UnimplementedUserServiceServer
    users map[string]*userv1.User
    mu    sync.RWMutex
}

func NewUserServer() *UserServer {
    return &UserServer{
        users: make(map[string]*userv1.User),
    }
}

// Unary RPC - Get User
func (s *UserServer) GetUser(ctx context.Context, req *userv1.GetUserRequest) (*userv1.GetUserResponse, error) {
    if req.Id == "" {
        return nil, status.Error(codes.InvalidArgument, "user ID is required")
    }

    s.mu.RLock()
    user, exists := s.users[req.Id]
    s.mu.RUnlock()

    if !exists {
        return nil, status.Error(codes.NotFound, "user not found")
    }

    return &userv1.GetUserResponse{
        User: user,
    }, nil
}

// Unary RPC - Create User
func (s *UserServer) CreateUser(ctx context.Context, req *userv1.CreateUserRequest) (*userv1.CreateUserResponse, error) {
    if req.Name == "" {
        return nil, status.Error(codes.InvalidArgument, "name is required")
    }
    if req.Email == "" {
        return nil, status.Error(codes.InvalidArgument, "email is required")
    }

    // Generate ID
    userID := fmt.Sprintf("user_%d", time.Now().UnixNano())
    now := timestamppb.Now()

    user := &userv1.User{
        Id:        userID,
        Name:      req.Name,
        Email:     req.Email,
        CreatedAt: now,
        UpdatedAt: now,
        Roles:     req.Roles,
        Metadata:  req.Metadata,
    }

    s.mu.Lock()
    s.users[userID] = user
    s.mu.Unlock()

    return &userv1.CreateUserResponse{
        User: user,
    }, nil
}

// Unary RPC - Update User
func (s *UserServer) UpdateUser(ctx context.Context, req *userv1.UpdateUserRequest) (*userv1.UpdateUserResponse, error) {
    if req.Id == "" {
        return nil, status.Error(codes.InvalidArgument, "user ID is required")
    }

    s.mu.Lock()
    defer s.mu.Unlock()

    user, exists := s.users[req.Id]
    if !exists {
        return nil, status.Error(codes.NotFound, "user not found")
    }

    // Update fields if provided
    if req.Name != nil {
        user.Name = *req.Name
    }
    if req.Email != nil {
        user.Email = *req.Email
    }
    if req.Roles != nil {
        user.Roles = req.Roles
    }
    if req.Metadata != nil {
        user.Metadata = req.Metadata
    }

    user.UpdatedAt = timestamppb.Now()

    return &userv1.UpdateUserResponse{
        User: user,
    }, nil
}

// Unary RPC - Delete User
func (s *UserServer) DeleteUser(ctx context.Context, req *userv1.DeleteUserRequest) (*emptypb.Empty, error) {
    if req.Id == "" {
        return nil, status.Error(codes.InvalidArgument, "user ID is required")
    }

    s.mu.Lock()
    defer s.mu.Unlock()

    if _, exists := s.users[req.Id]; !exists {
        return nil, status.Error(codes.NotFound, "user not found")
    }

    delete(s.users, req.Id)

    return &emptypb.Empty{}, nil
}

// Server Streaming RPC - List Users
func (s *UserServer) ListUsers(req *userv1.ListUsersRequest, stream userv1.UserService_ListUsersServer) error {
    s.mu.RLock()
    users := make([]*userv1.User, 0, len(s.users))
    for _, user := range s.users {
        users = append(users, user)
    }
    s.mu.RUnlock()

    // Apply pagination
    pageSize := int(req.PageSize)
    if pageSize <= 0 {
        pageSize = 10
    }

    for i, user := range users {
        if i >= pageSize {
            break
        }

        if err := stream.Send(user); err != nil {
            return err
        }
    }

    return nil
}

// Client Streaming RPC - Create Multiple Users
func (s *UserServer) CreateMultipleUsers(stream userv1.UserService_CreateMultipleUsersServer) error {
    var createdUsers []*userv1.User
    var createdCount int32

    for {
        req, err := stream.Recv()
        if err == io.EOF {
            // Client finished sending
            return stream.SendAndClose(&userv1.CreateMultipleUsersResponse{
                Users:        createdUsers,
                CreatedCount: createdCount,
            })
        }
        if err != nil {
            return err
        }

        // Create user
        userID := fmt.Sprintf("user_%d", time.Now().UnixNano())
        now := timestamppb.Now()

        user := &userv1.User{
            Id:        userID,
            Name:      req.Name,
            Email:     req.Email,
            CreatedAt: now,
            UpdatedAt: now,
            Roles:     req.Roles,
            Metadata:  req.Metadata,
        }

        s.mu.Lock()
        s.users[userID] = user
        s.mu.Unlock()

        createdUsers = append(createdUsers, user)
        createdCount++
    }
}

// Bidirectional Streaming RPC - Chat
func (s *UserServer) ChatWithUser(stream userv1.UserService_ChatWithUserServer) error {
    for {
        msg, err := stream.Recv()
        if err == io.EOF {
            return nil
        }
        if err != nil {
            return err
        }

        // Echo the message back with timestamp
        response := &userv1.ChatMessage{
            UserId:    msg.UserId,
            Message:   fmt.Sprintf("Echo: %s", msg.Message),
            Timestamp: timestamppb.Now(),
        }

        if err := stream.Send(response); err != nil {
            return err
        }
    }
}

// Middleware
func loggingInterceptor(
    ctx context.Context,
    req interface{},
    info *grpc.UnaryServerInfo,
    handler grpc.UnaryHandler,
) (interface{}, error) {
    start := time.Now()
    
    resp, err := handler(ctx, req)
    
    duration := time.Since(start)
    log.Printf("Method: %s, Duration: %v, Error: %v", info.FullMethod, duration, err)
    
    return resp, err
}

func main() {
    lis, err := net.Listen("tcp", ":50051")
    if err != nil {
        log.Fatalf("Failed to listen: %v", err)
    }

    s := grpc.NewServer(
        grpc.UnaryInterceptor(loggingInterceptor),
    )

    userServer := NewUserServer()
    userv1.RegisterUserServiceServer(s, userServer)

    log.Println("gRPC server listening on :50051")
    if err := s.Serve(lis); err != nil {
        log.Fatalf("Failed to serve: %v", err)
    }
}
```

### **gRPC Client Implementation (Go)**

```go
package main

import (
    "context"
    "io"
    "log"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
    
    userv1 "github.com/example/user/v1"
)

func main() {
    // Connect to server
    conn, err := grpc.Dial("localhost:50051", 
        grpc.WithTransportCredentials(insecure.NewCredentials()),
        grpc.WithTimeout(5*time.Second),
    )
    if err != nil {
        log.Fatalf("Failed to connect: %v", err)
    }
    defer conn.Close()

    client := userv1.NewUserServiceClient(conn)

    // Unary RPC example
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    // Create user
    createResp, err := client.CreateUser(ctx, &userv1.CreateUserRequest{
        Name:  "John Doe",
        Email: "john@example.com",
        Roles: []string{"user", "admin"},
        Metadata: map[string]string{
            "department": "engineering",
            "team":       "backend",
        },
    })
    if err != nil {
        log.Fatalf("CreateUser failed: %v", err)
    }
    log.Printf("Created user: %v", createResp.User)

    // Get user
    getResp, err := client.GetUser(ctx, &userv1.GetUserRequest{
        Id: createResp.User.Id,
    })
    if err != nil {
        log.Fatalf("GetUser failed: %v", err)
    }
    log.Printf("Retrieved user: %v", getResp.User)

    // Server streaming example
    listStream, err := client.ListUsers(ctx, &userv1.ListUsersRequest{
        PageSize: 10,
    })
    if err != nil {
        log.Fatalf("ListUsers failed: %v", err)
    }

    log.Println("Users:")
    for {
        user, err := listStream.Recv()
        if err == io.EOF {
            break
        }
        if err != nil {
            log.Fatalf("ListUsers stream error: %v", err)
        }
        log.Printf("- %s (%s)", user.Name, user.Email)
    }

    // Client streaming example
    createMultipleStream, err := client.CreateMultipleUsers(ctx)
    if err != nil {
        log.Fatalf("CreateMultipleUsers failed: %v", err)
    }

    users := []*userv1.CreateUserRequest{
        {Name: "Alice", Email: "alice@example.com"},
        {Name: "Bob", Email: "bob@example.com"},
        {Name: "Charlie", Email: "charlie@example.com"},
    }

    for _, user := range users {
        if err := createMultipleStream.Send(user); err != nil {
            log.Fatalf("Send failed: %v", err)
        }
    }

    createMultipleResp, err := createMultipleStream.CloseAndRecv()
    if err != nil {
        log.Fatalf("CloseAndRecv failed: %v", err)
    }
    log.Printf("Created %d users", createMultipleResp.CreatedCount)

    // Bidirectional streaming example
    chatStream, err := client.ChatWithUser(ctx)
    if err != nil {
        log.Fatalf("ChatWithUser failed: %v", err)
    }

    // Send messages
    go func() {
        messages := []string{"Hello", "How are you?", "Goodbye"}
        for _, msg := range messages {
            if err := chatStream.Send(&userv1.ChatMessage{
                UserId:  "user_123",
                Message: msg,
            }); err != nil {
                log.Printf("Send error: %v", err)
                return
            }
            time.Sleep(1 * time.Second)
        }
        chatStream.CloseSend()
    }()

    // Receive messages
    for {
        resp, err := chatStream.Recv()
        if err == io.EOF {
            break
        }
        if err != nil {
            log.Fatalf("Recv error: %v", err)
        }
        log.Printf("Received: %s", resp.Message)
    }
}
```

### **gRPC Advanced Features**

```go
// 1. Authentication and Authorization
func authInterceptor(
    ctx context.Context,
    req interface{},
    info *grpc.UnaryServerInfo,
    handler grpc.UnaryHandler,
) (interface{}, error) {
    // Extract token from metadata
    md, ok := metadata.FromIncomingContext(ctx)
    if !ok {
        return nil, status.Error(codes.Unauthenticated, "metadata not found")
    }

    tokens := md.Get("authorization")
    if len(tokens) == 0 {
        return nil, status.Error(codes.Unauthenticated, "authorization token not found")
    }

    token := strings.TrimPrefix(tokens[0], "Bearer ")
    
    // Validate token
    claims, err := validateJWT(token)
    if err != nil {
        return nil, status.Error(codes.Unauthenticated, "invalid token")
    }

    // Add user info to context
    ctx = context.WithValue(ctx, "user", claims)

    return handler(ctx, req)
}

// 2. Circuit Breaker Pattern
type CircuitBreaker struct {
    maxFailures int
    timeout     time.Duration
    failures    int
    lastFailure time.Time
    state       string // "closed", "open", "half-open"
    mu          sync.Mutex
}

func (cb *CircuitBreaker) Call(fn func() error) error {
    cb.mu.Lock()
    defer cb.mu.Unlock()

    if cb.state == "open" {
        if time.Since(cb.lastFailure) > cb.timeout {
            cb.state = "half-open"
        } else {
            return errors.New("circuit breaker is open")
        }
    }

    err := fn()
    
    if err != nil {
        cb.failures++
        cb.lastFailure = time.Now()
        
        if cb.failures >= cb.maxFailures {
            cb.state = "open"
        }
        
        return err
    }

    // Success
    cb.failures = 0
    cb.state = "closed"
    return nil
}

// 3. Health Check Service
type HealthService struct {
    healthpb.UnimplementedHealthServer
}

func (s *HealthService) Check(ctx context.Context, req *healthpb.HealthCheckRequest) (*healthpb.HealthCheckResponse, error) {
    return &healthpb.HealthCheckResponse{
        Status: healthpb.HealthCheckResponse_SERVING,
    }, nil
}

func (s *HealthService) Watch(req *healthpb.HealthCheckRequest, stream healthpb.Health_WatchServer) error {
    // Implementation for health status streaming
    return nil
}

// 4. Load Balancing and Service Discovery
func setupClientWithLoadBalancing() *grpc.ClientConn {
    // Register resolver
    resolver.Register(&consulResolver{})
    
    conn, err := grpc.Dial(
        "consul://localhost:8500/user-service",
        grpc.WithTransportCredentials(insecure.NewCredentials()),
        grpc.WithDefaultServiceConfig(`{
            "loadBalancingPolicy": "round_robin",
            "healthCheckConfig": {
                "serviceName": "user-service"
            }
        }`),
    )
    
    if err != nil {
        log.Fatal(err)
    }
    
    return conn
}
```

---

## 🌊 **Event Streaming**

### **Apache Kafka with Go**

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "time"

    "github.com/segmentio/kafka-go"
)

// Event structures
type PaymentEvent struct {
    ID          string    `json:"id"`
    Type        string    `json:"type"`
    UserID      string    `json:"user_id"`
    Amount      float64   `json:"amount"`
    Currency    string    `json:"currency"`
    Status      string    `json:"status"`
    Timestamp   time.Time `json:"timestamp"`
    Metadata    map[string]interface{} `json:"metadata"`
}

type UserEvent struct {
    ID        string    `json:"id"`
    Type      string    `json:"type"`
    UserID    string    `json:"user_id"`
    Action    string    `json:"action"`
    Timestamp time.Time `json:"timestamp"`
    Data      map[string]interface{} `json:"data"`
}

// Kafka Producer
type EventProducer struct {
    writer *kafka.Writer
}

func NewEventProducer(brokers []string, topic string) *EventProducer {
    writer := &kafka.Writer{
        Addr:         kafka.TCP(brokers...),
        Topic:        topic,
        Balancer:     &kafka.LeastBytes{},
        RequiredAcks: kafka.RequireAll,
        Async:        false,
        Compression:  kafka.Snappy,
        BatchSize:    100,
        BatchTimeout: 10 * time.Millisecond,
    }

    return &EventProducer{writer: writer}
}

func (p *EventProducer) PublishPaymentEvent(ctx context.Context, event PaymentEvent) error {
    eventData, err := json.Marshal(event)
    if err != nil {
        return fmt.Errorf("failed to marshal event: %w", err)
    }

    message := kafka.Message{
        Key:   []byte(event.ID),
        Value: eventData,
        Headers: []kafka.Header{
            {Key: "event-type", Value: []byte(event.Type)},
            {Key: "user-id", Value: []byte(event.UserID)},
            {Key: "timestamp", Value: []byte(event.Timestamp.Format(time.RFC3339))},
        },
    }

    err = p.writer.WriteMessages(ctx, message)
    if err != nil {
        return fmt.Errorf("failed to write message: %w", err)
    }

    log.Printf("Published payment event: %s", event.ID)
    return nil
}

func (p *EventProducer) PublishBatchEvents(ctx context.Context, events []PaymentEvent) error {
    messages := make([]kafka.Message, len(events))
    
    for i, event := range events {
        eventData, err := json.Marshal(event)
        if err != nil {
            return fmt.Errorf("failed to marshal event %s: %w", event.ID, err)
        }

        messages[i] = kafka.Message{
            Key:   []byte(event.ID),
            Value: eventData,
            Headers: []kafka.Header{
                {Key: "event-type", Value: []byte(event.Type)},
                {Key: "user-id", Value: []byte(event.UserID)},
            },
        }
    }

    err := p.writer.WriteMessages(ctx, messages...)
    if err != nil {
        return fmt.Errorf("failed to write batch messages: %w", err)
    }

    log.Printf("Published %d events in batch", len(events))
    return nil
}

func (p *EventProducer) Close() error {
    return p.writer.Close()
}

// Kafka Consumer
type EventConsumer struct {
    reader *kafka.Reader
}

func NewEventConsumer(brokers []string, topic, groupID string) *EventConsumer {
    reader := kafka.NewReader(kafka.ReaderConfig{
        Brokers:  brokers,
        Topic:    topic,
        GroupID:  groupID,
        MinBytes: 10e3, // 10KB
        MaxBytes: 10e6, // 10MB
        MaxWait:  1 * time.Second,
        CommitInterval: 1 * time.Second,
    })

    return &EventConsumer{reader: reader}
}

func (c *EventConsumer) ProcessPaymentEvents(ctx context.Context, handler func(PaymentEvent) error) error {
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
            message, err := c.reader.ReadMessage(ctx)
            if err != nil {
                log.Printf("Error reading message: %v", err)
                continue
            }

            var event PaymentEvent
            if err := json.Unmarshal(message.Value, &event); err != nil {
                log.Printf("Error unmarshaling event: %v", err)
                continue
            }

            // Process event
            if err := handler(event); err != nil {
                log.Printf("Error processing event %s: %v", event.ID, err)
                // Implement retry logic or dead letter queue
                continue
            }

            log.Printf("Processed payment event: %s", event.ID)
        }
    }
}

func (c *EventConsumer) Close() error {
    return c.reader.Close()
}

// Event Processor with Error Handling
type EventProcessor struct {
    consumer    *EventConsumer
    retryQueue  chan kafka.Message
    dlq         *EventProducer
    maxRetries  int
}

func NewEventProcessor(consumer *EventConsumer, dlq *EventProducer, maxRetries int) *EventProcessor {
    return &EventProcessor{
        consumer:   consumer,
        retryQueue: make(chan kafka.Message, 1000),
        dlq:        dlq,
        maxRetries: maxRetries,
    }
}

func (ep *EventProcessor) processWithRetry(ctx context.Context, event PaymentEvent, retryCount int) error {
    // Business logic
    if err := ep.processPaymentEvent(event); err != nil {
        if retryCount < ep.maxRetries {
            // Retry with exponential backoff
            backoff := time.Duration(retryCount*retryCount) * time.Second
            time.Sleep(backoff)
            return ep.processWithRetry(ctx, event, retryCount+1)
        } else {
            // Send to Dead Letter Queue
            return ep.sendToDLQ(ctx, event)
        }
    }
    
    return nil
}

func (ep *EventProcessor) processPaymentEvent(event PaymentEvent) error {
    // Simulate processing
    switch event.Type {
    case "payment_created":
        return ep.handlePaymentCreated(event)
    case "payment_completed":
        return ep.handlePaymentCompleted(event)
    case "payment_failed":
        return ep.handlePaymentFailed(event)
    default:
        return fmt.Errorf("unknown event type: %s", event.Type)
    }
}

func (ep *EventProcessor) handlePaymentCreated(event PaymentEvent) error {
    log.Printf("Processing payment created: %s for user %s", event.ID, event.UserID)
    
    // Update database
    // Send notifications
    // Update analytics
    
    return nil
}

func (ep *EventProcessor) handlePaymentCompleted(event PaymentEvent) error {
    log.Printf("Processing payment completed: %s", event.ID)
    
    // Update payment status
    // Send confirmation email
    // Update user balance
    // Trigger loyalty points
    
    return nil
}

func (ep *EventProcessor) handlePaymentFailed(event PaymentEvent) error {
    log.Printf("Processing payment failed: %s", event.ID)
    
    // Update payment status
    // Send failure notification
    // Log for investigation
    
    return nil
}

func (ep *EventProcessor) sendToDLQ(ctx context.Context, event PaymentEvent) error {
    log.Printf("Sending event %s to DLQ", event.ID)
    return ep.dlq.PublishPaymentEvent(ctx, event)
}

// Kafka Streams-like Processing
type StreamProcessor struct {
    inputTopic  string
    outputTopic string
    producer    *EventProducer
    consumer    *EventConsumer
}

func NewStreamProcessor(inputTopic, outputTopic string, brokers []string) *StreamProcessor {
    producer := NewEventProducer(brokers, outputTopic)
    consumer := NewEventConsumer(brokers, inputTopic, "stream-processor")
    
    return &StreamProcessor{
        inputTopic:  inputTopic,
        outputTopic: outputTopic,
        producer:    producer,
        consumer:    consumer,
    }
}

func (sp *StreamProcessor) ProcessStream(ctx context.Context) error {
    return sp.consumer.ProcessPaymentEvents(ctx, func(event PaymentEvent) error {
        // Transform event
        transformedEvent := sp.transformEvent(event)
        
        // Enrich event
        enrichedEvent, err := sp.enrichEvent(transformedEvent)
        if err != nil {
            return err
        }
        
        // Publish to output topic
        return sp.producer.PublishPaymentEvent(ctx, enrichedEvent)
    })
}

func (sp *StreamProcessor) transformEvent(event PaymentEvent) PaymentEvent {
    // Apply transformations
    event.Amount = event.Amount / 100 // Convert from cents
    return event
}

func (sp *StreamProcessor) enrichEvent(event PaymentEvent) (PaymentEvent, error) {
    // Enrich with additional data
    if event.Metadata == nil {
        event.Metadata = make(map[string]interface{})
    }
    
    event.Metadata["processed_at"] = time.Now()
    event.Metadata["processor"] = "stream-processor"
    
    return event, nil
}

// Event Sourcing Pattern
type EventStore struct {
    producer *EventProducer
}

func NewEventStore(brokers []string) *EventStore {
    producer := NewEventProducer(brokers, "event-store")
    return &EventStore{producer: producer}
}

func (es *EventStore) AppendEvent(ctx context.Context, aggregateID string, event interface{}) error {
    eventData, err := json.Marshal(event)
    if err != nil {
        return err
    }

    kafkaEvent := PaymentEvent{
        ID:        fmt.Sprintf("%s-%d", aggregateID, time.Now().UnixNano()),
        Type:      "domain_event",
        UserID:    aggregateID,
        Timestamp: time.Now(),
        Metadata: map[string]interface{}{
            "aggregate_id":   aggregateID,
            "event_data":     json.RawMessage(eventData),
            "event_version":  1,
        },
    }

    return es.producer.PublishPaymentEvent(ctx, kafkaEvent)
}

// CQRS Pattern Implementation
type CommandHandler struct {
    eventStore *EventStore
}

func NewCommandHandler(eventStore *EventStore) *CommandHandler {
    return &CommandHandler{eventStore: eventStore}
}

type CreatePaymentCommand struct {
    PaymentID string  `json:"payment_id"`
    UserID    string  `json:"user_id"`
    Amount    float64 `json:"amount"`
    Currency  string  `json:"currency"`
}

func (ch *CommandHandler) HandleCreatePayment(ctx context.Context, cmd CreatePaymentCommand) error {
    // Business logic validation
    if cmd.Amount <= 0 {
        return fmt.Errorf("invalid amount: %f", cmd.Amount)
    }

    // Create domain event
    event := map[string]interface{}{
        "event_type":  "PaymentCreated",
        "payment_id":  cmd.PaymentID,
        "user_id":     cmd.UserID,
        "amount":      cmd.Amount,
        "currency":    cmd.Currency,
        "created_at":  time.Now(),
    }

    // Append to event store
    return ch.eventStore.AppendEvent(ctx, cmd.PaymentID, event)
}

// Main application
func main() {
    brokers := []string{"localhost:9092"}
    
    // Setup producer
    producer := NewEventProducer(brokers, "payment-events")
    defer producer.Close()

    // Setup consumer
    consumer := NewEventConsumer(brokers, "payment-events", "payment-processor")
    defer consumer.Close()

    // Setup DLQ
    dlq := NewEventProducer(brokers, "payment-events-dlq")
    defer dlq.Close()

    // Setup event processor
    processor := NewEventProcessor(consumer, dlq, 3)

    ctx := context.Background()

    // Publish sample events
    go func() {
        for i := 0; i < 10; i++ {
            event := PaymentEvent{
                ID:        fmt.Sprintf("payment_%d", i),
                Type:      "payment_created",
                UserID:    fmt.Sprintf("user_%d", i%3),
                Amount:    float64(100 + i*10),
                Currency:  "USD",
                Status:    "pending",
                Timestamp: time.Now(),
                Metadata: map[string]interface{}{
                    "source": "api",
                    "version": "1.0",
                },
            }

            if err := producer.PublishPaymentEvent(ctx, event); err != nil {
                log.Printf("Error publishing event: %v", err)
            }

            time.Sleep(1 * time.Second)
        }
    }()

    // Process events
    go func() {
        err := consumer.ProcessPaymentEvents(ctx, func(event PaymentEvent) error {
            return processor.processWithRetry(ctx, event, 0)
        })
        if err != nil {
            log.Printf("Consumer error: %v", err)
        }
    }()

    // Keep running
    select {}
}
```

### **Event Streaming Best Practices**

```go
// 1. Schema Registry Integration
type SchemaRegistry struct {
    baseURL string
    client  *http.Client
}

func NewSchemaRegistry(baseURL string) *SchemaRegistry {
    return &SchemaRegistry{
        baseURL: baseURL,
        client:  &http.Client{Timeout: 10 * time.Second},
    }
}

func (sr *SchemaRegistry) ValidateEvent(event interface{}, schemaVersion int) error {
    // Validate event against schema
    return nil
}

// 2. Event Deduplication
type DeduplicationCache struct {
    cache map[string]time.Time
    mu    sync.RWMutex
    ttl   time.Duration
}

func NewDeduplicationCache(ttl time.Duration) *DeduplicationCache {
    return &DeduplicationCache{
        cache: make(map[string]time.Time),
        ttl:   ttl,
    }
}

func (dc *DeduplicationCache) IsDuplicate(eventID string) bool {
    dc.mu.RLock()
    defer dc.mu.RUnlock()
    
    timestamp, exists := dc.cache[eventID]
    if !exists {
        return false
    }
    
    return time.Since(timestamp) < dc.ttl
}

func (dc *DeduplicationCache) AddEvent(eventID string) {
    dc.mu.Lock()
    defer dc.mu.Unlock()
    
    dc.cache[eventID] = time.Now()
}

// 3. Monitoring and Metrics
type EventMetrics struct {
    eventsProduced   prometheus.Counter
    eventsConsumed   prometheus.Counter
    processingTime   prometheus.Histogram
    errorRate        prometheus.Counter
}

func NewEventMetrics() *EventMetrics {
    return &EventMetrics{
        eventsProduced: prometheus.NewCounter(prometheus.CounterOpts{
            Name: "events_produced_total",
            Help: "Total number of events produced",
        }),
        eventsConsumed: prometheus.NewCounter(prometheus.CounterOpts{
            Name: "events_consumed_total", 
            Help: "Total number of events consumed",
        }),
        processingTime: prometheus.NewHistogram(prometheus.HistogramOpts{
            Name: "event_processing_duration_seconds",
            Help: "Time spent processing events",
        }),
        errorRate: prometheus.NewCounter(prometheus.CounterOpts{
            Name: "event_processing_errors_total",
            Help: "Total number of event processing errors",
        }),
    }
}
```

---

## 🎯 **Interview Questions**

### **GraphQL Questions**

**Q1: What are the main advantages and disadvantages of GraphQL over REST?**

**Answer:**

**Advantages:**
- **Single endpoint**: One URL for all data needs
- **Precise data fetching**: Request exactly what you need
- **Strong type system**: Self-documenting with schema
- **Real-time subscriptions**: Built-in WebSocket support
- **No over/under-fetching**: Eliminates N+1 problems

**Disadvantages:**
- **Complexity**: Harder to implement than REST
- **Caching challenges**: HTTP caching doesn't work well
- **Query complexity**: Need to limit query depth/cost
- **Learning curve**: Teams need GraphQL expertise

**Q2: How do you solve the N+1 problem in GraphQL?**

**Answer:**
```javascript
// Problem: N+1 queries
const resolvers = {
    Post: {
        author: (post) => User.findById(post.authorId) // N+1 problem
    }
};

// Solution: DataLoader
const DataLoader = require('dataloader');

const userLoader = new DataLoader(async (userIds) => {
    const users = await User.findByIds(userIds);
    return userIds.map(id => users.find(user => user.id === id));
});

const resolvers = {
    Post: {
        author: (post) => userLoader.load(post.authorId) // Batched
    }
};
```

### **gRPC Questions**

**Q3: When would you choose gRPC over REST?**

**Answer:**

**Choose gRPC when:**
- **Performance critical**: Binary protocol is faster
- **Type safety**: Strong typing with Protocol Buffers
- **Microservices**: Internal service communication
- **Streaming**: Real-time bidirectional communication
- **Multiple languages**: Language-agnostic interface

**Choose REST when:**
- **Browser compatibility**: Direct web client access
- **Public APIs**: Better developer experience
- **Debugging**: Human-readable HTTP
- **Caching**: HTTP caching infrastructure

**Q4: How do you handle errors in gRPC?**

**Answer:**
```go
// Custom error with details
st := status.New(codes.InvalidArgument, "invalid user data")
st, _ = st.WithDetails(&errdetails.BadRequest{
    FieldViolations: []*errdetails.BadRequest_FieldViolation{
        {
            Field:       "email",
            Description: "email format is invalid",
        },
    },
})
return st.Err()

// Client-side error handling
if err != nil {
    if st, ok := status.FromError(err); ok {
        switch st.Code() {
        case codes.InvalidArgument:
            // Handle validation errors
        case codes.NotFound:
            // Handle not found
        case codes.Unauthenticated:
            // Handle auth errors
        }
    }
}
```

### **Event Streaming Questions**

**Q5: How do you ensure exactly-once delivery in event streaming?**

**Answer:**

**Approaches:**
1. **Idempotent processing**: Make operations idempotent
2. **Deduplication**: Track processed event IDs
3. **Transactional outbox**: Combine DB writes with event publishing
4. **Kafka transactions**: Use Kafka's exactly-once semantics

```go
// Idempotent processing example
func ProcessPayment(event PaymentEvent) error {
    // Check if already processed
    if isProcessed(event.ID) {
        return nil // Already processed
    }
    
    // Process atomically
    tx := db.Begin()
    defer tx.Rollback()
    
    if err := updatePaymentStatus(tx, event.ID, "completed"); err != nil {
        return err
    }
    
    if err := markAsProcessed(tx, event.ID); err != nil {
        return err
    }
    
    return tx.Commit()
}
```

**Q6: Design an event-driven architecture for a payment system.**

**Answer:**
```
Services:
- Payment Service: Handles payment processing
- User Service: Manages user accounts  
- Notification Service: Sends emails/SMS
- Analytics Service: Tracks metrics

Events:
- PaymentCreated
- PaymentCompleted
- PaymentFailed
- UserUpdated

Flow:
1. Payment Service publishes PaymentCreated
2. User Service updates balance (if needed)
3. Notification Service sends confirmation
4. Analytics Service updates metrics

Benefits:
- Loose coupling between services
- Scalable event processing
- Audit trail of all events
- Easy to add new services
```

---

## 🚀 **Getting Started Checklist**

### **GraphQL**
- [ ] Understand schema design principles
- [ ] Learn resolver patterns and DataLoader
- [ ] Implement query complexity analysis
- [ ] Set up subscriptions for real-time data
- [ ] Add proper error handling and validation

### **gRPC**
- [ ] Learn Protocol Buffers syntax
- [ ] Understand different RPC types
- [ ] Implement proper error handling
- [ ] Add authentication and authorization
- [ ] Set up load balancing and service discovery

### **Event Streaming**
- [ ] Design event schemas and naming conventions
- [ ] Implement producer and consumer patterns
- [ ] Add error handling and retry logic
- [ ] Set up monitoring and alerting
- [ ] Understand consistency patterns (eventual vs strong)

### **General Best Practices**
- [ ] Design for failure and implement circuit breakers
- [ ] Add comprehensive monitoring and metrics
- [ ] Implement proper security (authentication, authorization)
- [ ] Consider schema evolution and versioning
- [ ] Plan for scaling and performance optimization

---

This comprehensive guide covers the essential modern technologies that are increasingly important in backend engineering interviews. Understanding these technologies and their implementation patterns will significantly strengthen your technical interview performance.