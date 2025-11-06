---
# Auto-generated front matter
Title: Grpc Complete Guide
LastUpdated: 2025-11-06T20:45:58.290719
Tags: []
Status: draft
---

# 🚀 gRPC Complete Guide

> **Comprehensive guide to gRPC for microservices communication and API development**

## 📚 Table of Contents

1. [Introduction to gRPC](#-introduction-to-grpc)
2. [Protocol Buffers](#-protocol-buffers)
3. [gRPC Service Definition](#-grpc-service-definition)
4. [gRPC Communication Patterns](#-grpc-communication-patterns)
5. [Go gRPC Implementation](#-go-grpc-implementation)
6. [Error Handling](#-error-handling)
7. [Authentication & Security](#-authentication--security)
8. [Load Balancing](#-load-balancing)
9. [Monitoring & Observability](#-monitoring--observability)
10. [Best Practices](#-best-practices)
11. [Real-world Examples](#-real-world-examples)

---

## 🌟 Introduction to gRPC

### What is gRPC?

gRPC (gRPC Remote Procedure Calls) is a high-performance, open-source RPC framework developed by Google. It uses HTTP/2 for transport, Protocol Buffers as the interface definition language, and provides features such as authentication, load balancing, and more.

### Key Features

- **High Performance**: Uses HTTP/2 and Protocol Buffers for efficient communication
- **Language Agnostic**: Supports multiple programming languages
- **Streaming**: Supports unary, server streaming, client streaming, and bidirectional streaming
- **Code Generation**: Automatic client and server code generation
- **Built-in Features**: Authentication, load balancing, health checking, and more

### gRPC vs REST

| Feature | gRPC | REST |
|---------|------|------|
| Protocol | HTTP/2 | HTTP/1.1/2 |
| Data Format | Protocol Buffers | JSON/XML |
| Performance | High | Medium |
| Streaming | Native | Limited |
| Code Generation | Yes | No |
| Browser Support | Limited | Full |

---

## 📦 Protocol Buffers

### What are Protocol Buffers?

Protocol Buffers (protobuf) is a language-neutral, platform-neutral, extensible mechanism for serializing structured data. It's used as the interface definition language for gRPC.

### Basic Syntax

```protobuf
syntax = "proto3";

package user;

option go_package = "github.com/example/user/pb";

// User service definition
service UserService {
  rpc GetUser(GetUserRequest) returns (GetUserResponse);
  rpc CreateUser(CreateUserRequest) returns (CreateUserResponse);
  rpc UpdateUser(UpdateUserRequest) returns (UpdateUserResponse);
  rpc DeleteUser(DeleteUserRequest) returns (DeleteUserResponse);
  rpc ListUsers(ListUsersRequest) returns (stream User);
}

// Message definitions
message User {
  int32 id = 1;
  string name = 2;
  string email = 3;
  int64 created_at = 4;
  UserStatus status = 5;
}

message GetUserRequest {
  int32 id = 1;
}

message GetUserResponse {
  User user = 1;
}

message CreateUserRequest {
  string name = 1;
  string email = 2;
}

message CreateUserResponse {
  User user = 1;
}

message UpdateUserRequest {
  int32 id = 1;
  string name = 2;
  string email = 3;
}

message UpdateUserResponse {
  User user = 1;
}

message DeleteUserRequest {
  int32 id = 1;
}

message DeleteUserResponse {
  bool success = 1;
}

message ListUsersRequest {
  int32 page = 1;
  int32 page_size = 2;
}

// Enums
enum UserStatus {
  USER_STATUS_UNSPECIFIED = 0;
  USER_STATUS_ACTIVE = 1;
  USER_STATUS_INACTIVE = 2;
  USER_STATUS_SUSPENDED = 3;
}
```

### Advanced Features

```protobuf
syntax = "proto3";

package advanced;

import "google/protobuf/timestamp.proto";
import "google/protobuf/empty.proto";

// Oneof fields
message PaymentRequest {
  oneof payment_method {
    CreditCard credit_card = 1;
    BankAccount bank_account = 2;
    DigitalWallet digital_wallet = 3;
  }
}

message CreditCard {
  string number = 1;
  string expiry = 2;
  string cvv = 3;
}

message BankAccount {
  string account_number = 1;
  string routing_number = 2;
}

message DigitalWallet {
  string wallet_id = 1;
  string provider = 2;
}

// Nested messages
message Address {
  string street = 1;
  string city = 2;
  string state = 3;
  string zip_code = 4;
  string country = 5;
}

message UserProfile {
  int32 id = 1;
  string name = 2;
  string email = 3;
  Address address = 4;
  repeated string phone_numbers = 5;
  map<string, string> metadata = 6;
  google.protobuf.Timestamp created_at = 7;
}

// Service with streaming
service PaymentService {
  rpc ProcessPayment(PaymentRequest) returns (PaymentResponse);
  rpc StreamPayments(google.protobuf.Empty) returns (stream PaymentResponse);
  rpc ProcessPaymentStream(stream PaymentRequest) returns (PaymentResponse);
  rpc BidirectionalPaymentStream(stream PaymentRequest) returns (stream PaymentResponse);
}
```

---

## 🔧 gRPC Service Definition

### Service Types

#### 1. Unary RPC
```protobuf
service UserService {
  rpc GetUser(GetUserRequest) returns (GetUserResponse);
}
```

#### 2. Server Streaming RPC
```protobuf
service UserService {
  rpc ListUsers(ListUsersRequest) returns (stream User);
}
```

#### 3. Client Streaming RPC
```protobuf
service UserService {
  rpc CreateUsers(stream CreateUserRequest) returns (CreateUsersResponse);
}
```

#### 4. Bidirectional Streaming RPC
```protobuf
service UserService {
  rpc ChatWithUsers(stream ChatMessage) returns (stream ChatMessage);
}
```

### Complete Service Definition

```protobuf
syntax = "proto3";

package ecommerce;

option go_package = "github.com/example/ecommerce/pb";

import "google/protobuf/timestamp.proto";
import "google/protobuf/empty.proto";

// Product service
service ProductService {
  rpc GetProduct(GetProductRequest) returns (GetProductResponse);
  rpc ListProducts(ListProductsRequest) returns (stream Product);
  rpc CreateProduct(CreateProductRequest) returns (CreateProductResponse);
  rpc UpdateProduct(UpdateProductRequest) returns (UpdateProductResponse);
  rpc DeleteProduct(DeleteProductRequest) returns (google.protobuf.Empty);
  rpc SearchProducts(SearchProductsRequest) returns (stream Product);
}

// Order service
service OrderService {
  rpc CreateOrder(CreateOrderRequest) returns (CreateOrderResponse);
  rpc GetOrder(GetOrderRequest) returns (GetOrderResponse);
  rpc UpdateOrderStatus(UpdateOrderStatusRequest) returns (UpdateOrderStatusResponse);
  rpc ListOrders(ListOrdersRequest) returns (stream Order);
  rpc StreamOrderUpdates(GetOrderRequest) returns (stream OrderUpdate);
}

// Payment service
service PaymentService {
  rpc ProcessPayment(PaymentRequest) returns (PaymentResponse);
  rpc RefundPayment(RefundRequest) returns (RefundResponse);
  rpc GetPaymentHistory(GetPaymentHistoryRequest) returns (stream Payment);
}

// Message definitions
message Product {
  int32 id = 1;
  string name = 2;
  string description = 3;
  double price = 4;
  int32 stock = 5;
  ProductCategory category = 6;
  repeated string tags = 7;
  google.protobuf.Timestamp created_at = 8;
  google.protobuf.Timestamp updated_at = 9;
}

message Order {
  int32 id = 1;
  int32 user_id = 2;
  repeated OrderItem items = 3;
  double total_amount = 4;
  OrderStatus status = 5;
  Address shipping_address = 6;
  google.protobuf.Timestamp created_at = 7;
  google.protobuf.Timestamp updated_at = 8;
}

message OrderItem {
  int32 product_id = 1;
  string product_name = 2;
  int32 quantity = 3;
  double unit_price = 4;
  double total_price = 5;
}

message Payment {
  int32 id = 1;
  int32 order_id = 2;
  double amount = 3;
  PaymentMethod method = 4;
  PaymentStatus status = 5;
  string transaction_id = 6;
  google.protobuf.Timestamp processed_at = 7;
}

message Address {
  string street = 1;
  string city = 2;
  string state = 3;
  string zip_code = 4;
  string country = 5;
}

// Request/Response messages
message GetProductRequest {
  int32 id = 1;
}

message GetProductResponse {
  Product product = 1;
}

message ListProductsRequest {
  int32 page = 1;
  int32 page_size = 2;
  ProductCategory category = 3;
}

message CreateProductRequest {
  string name = 1;
  string description = 2;
  double price = 3;
  int32 stock = 4;
  ProductCategory category = 5;
  repeated string tags = 6;
}

message CreateProductResponse {
  Product product = 1;
}

message UpdateProductRequest {
  int32 id = 1;
  string name = 2;
  string description = 3;
  double price = 4;
  int32 stock = 5;
  ProductCategory category = 6;
  repeated string tags = 7;
}

message UpdateProductResponse {
  Product product = 1;
}

message DeleteProductRequest {
  int32 id = 1;
}

message SearchProductsRequest {
  string query = 1;
  ProductCategory category = 2;
  double min_price = 3;
  double max_price = 4;
}

message CreateOrderRequest {
  int32 user_id = 1;
  repeated OrderItem items = 2;
  Address shipping_address = 3;
}

message CreateOrderResponse {
  Order order = 1;
}

message GetOrderRequest {
  int32 id = 1;
}

message GetOrderResponse {
  Order order = 1;
}

message UpdateOrderStatusRequest {
  int32 id = 1;
  OrderStatus status = 2;
}

message UpdateOrderStatusResponse {
  Order order = 1;
}

message ListOrdersRequest {
  int32 user_id = 1;
  OrderStatus status = 2;
  int32 page = 3;
  int32 page_size = 4;
}

message OrderUpdate {
  int32 order_id = 1;
  OrderStatus status = 2;
  string message = 3;
  google.protobuf.Timestamp timestamp = 4;
}

message PaymentRequest {
  int32 order_id = 1;
  double amount = 2;
  PaymentMethod method = 3;
  string card_number = 4;
  string expiry_date = 5;
  string cvv = 6;
}

message PaymentResponse {
  Payment payment = 1;
  bool success = 2;
  string message = 3;
}

message RefundRequest {
  int32 payment_id = 1;
  double amount = 2;
  string reason = 3;
}

message RefundResponse {
  bool success = 1;
  string message = 2;
  string refund_id = 3;
}

message GetPaymentHistoryRequest {
  int32 user_id = 1;
  int32 page = 2;
  int32 page_size = 3;
}

// Enums
enum ProductCategory {
  PRODUCT_CATEGORY_UNSPECIFIED = 0;
  PRODUCT_CATEGORY_ELECTRONICS = 1;
  PRODUCT_CATEGORY_CLOTHING = 2;
  PRODUCT_CATEGORY_BOOKS = 3;
  PRODUCT_CATEGORY_HOME = 4;
  PRODUCT_CATEGORY_SPORTS = 5;
}

enum OrderStatus {
  ORDER_STATUS_UNSPECIFIED = 0;
  ORDER_STATUS_PENDING = 1;
  ORDER_STATUS_CONFIRMED = 2;
  ORDER_STATUS_SHIPPED = 3;
  ORDER_STATUS_DELIVERED = 4;
  ORDER_STATUS_CANCELLED = 5;
  ORDER_STATUS_REFUNDED = 6;
}

enum PaymentMethod {
  PAYMENT_METHOD_UNSPECIFIED = 0;
  PAYMENT_METHOD_CREDIT_CARD = 1;
  PAYMENT_METHOD_DEBIT_CARD = 2;
  PAYMENT_METHOD_PAYPAL = 3;
  PAYMENT_METHOD_BANK_TRANSFER = 4;
  PAYMENT_METHOD_DIGITAL_WALLET = 5;
}

enum PaymentStatus {
  PAYMENT_STATUS_UNSPECIFIED = 0;
  PAYMENT_STATUS_PENDING = 1;
  PAYMENT_STATUS_PROCESSING = 2;
  PAYMENT_STATUS_COMPLETED = 3;
  PAYMENT_STATUS_FAILED = 4;
  PAYMENT_STATUS_REFUNDED = 5;
}
```

---

## 🔄 gRPC Communication Patterns

### 1. Unary RPC

```go
package main

import (
    "context"
    "log"
    "net"

    "google.golang.org/grpc"
    pb "github.com/example/ecommerce/pb"
)

type ProductServer struct {
    pb.UnimplementedProductServiceServer
    products map[int32]*pb.Product
}

func (s *ProductServer) GetProduct(ctx context.Context, req *pb.GetProductRequest) (*pb.GetProductResponse, error) {
    product, exists := s.products[req.Id]
    if !exists {
        return nil, grpc.Errorf(grpc.Code(codes.NotFound), "Product not found")
    }
    
    return &pb.GetProductResponse{Product: product}, nil
}

func (s *ProductServer) CreateProduct(ctx context.Context, req *pb.CreateProductRequest) (*pb.CreateProductResponse, error) {
    // Generate new ID
    id := int32(len(s.products) + 1)
    
    product := &pb.Product{
        Id:          id,
        Name:        req.Name,
        Description: req.Description,
        Price:       req.Price,
        Stock:       req.Stock,
        Category:    req.Category,
        Tags:        req.Tags,
    }
    
    s.products[id] = product
    
    return &pb.CreateProductResponse{Product: product}, nil
}

func main() {
    lis, err := net.Listen("tcp", ":50051")
    if err != nil {
        log.Fatalf("Failed to listen: %v", err)
    }
    
    s := grpc.NewServer()
    pb.RegisterProductServiceServer(s, &ProductServer{
        products: make(map[int32]*pb.Product),
    })
    
    log.Println("Server starting on :50051")
    if err := s.Serve(lis); err != nil {
        log.Fatalf("Failed to serve: %v", err)
    }
}
```

### 2. Server Streaming RPC

```go
func (s *ProductServer) ListProducts(req *pb.ListProductsRequest, stream pb.ProductService_ListProductsServer) error {
    for _, product := range s.products {
        // Apply filters
        if req.Category != pb.ProductCategory_PRODUCT_CATEGORY_UNSPECIFIED && 
           product.Category != req.Category {
            continue
        }
        
        // Send product to client
        if err := stream.Send(product); err != nil {
            return err
        }
    }
    
    return nil
}

func (s *ProductServer) SearchProducts(req *pb.SearchProductsRequest, stream pb.ProductService_SearchProductsServer) error {
    for _, product := range s.products {
        // Simple search implementation
        if strings.Contains(strings.ToLower(product.Name), strings.ToLower(req.Query)) ||
           strings.Contains(strings.ToLower(product.Description), strings.ToLower(req.Query)) {
            
            // Apply price filters
            if req.MinPrice > 0 && product.Price < req.MinPrice {
                continue
            }
            if req.MaxPrice > 0 && product.Price > req.MaxPrice {
                continue
            }
            
            if err := stream.Send(product); err != nil {
                return err
            }
        }
    }
    
    return nil
}
```

### 3. Client Streaming RPC

```go
func (s *ProductServer) CreateProducts(stream pb.ProductService_CreateProductsServer) error {
    var products []*pb.Product
    var totalPrice float64
    
    for {
        req, err := stream.Recv()
        if err == io.EOF {
            // All products received, send response
            return stream.SendAndClose(&pb.CreateProductsResponse{
                Products:    products,
                TotalPrice:  totalPrice,
                Count:       int32(len(products)),
            })
        }
        if err != nil {
            return err
        }
        
        // Create product
        id := int32(len(s.products) + 1)
        product := &pb.Product{
            Id:          id,
            Name:        req.Name,
            Description: req.Description,
            Price:       req.Price,
            Stock:       req.Stock,
            Category:    req.Category,
            Tags:        req.Tags,
        }
        
        s.products[id] = product
        products = append(products, product)
        totalPrice += product.Price
    }
}
```

### 4. Bidirectional Streaming RPC

```go
func (s *ProductServer) ChatWithProducts(stream pb.ProductService_ChatWithProductsServer) error {
    for {
        req, err := stream.Recv()
        if err == io.EOF {
            return nil
        }
        if err != nil {
            return err
        }
        
        // Process chat message
        response := &pb.ChatMessage{
            Id:        req.Id,
            Message:   "Product information: " + req.Message,
            Timestamp: time.Now().Unix(),
        }
        
        // Send response back to client
        if err := stream.Send(response); err != nil {
            return err
        }
    }
}
```

---

## 🚀 Go gRPC Implementation

### Client Implementation

```go
package main

import (
    "context"
    "log"
    "time"

    "google.golang.org/grpc"
    pb "github.com/example/ecommerce/pb"
)

func main() {
    // Connect to server
    conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
    if err != nil {
        log.Fatalf("Failed to connect: %v", err)
    }
    defer conn.Close()
    
    // Create client
    client := pb.NewProductServiceClient(conn)
    
    // Unary RPC
    ctx, cancel := context.WithTimeout(context.Background(), time.Second)
    defer cancel()
    
    // Get product
    resp, err := client.GetProduct(ctx, &pb.GetProductRequest{Id: 1})
    if err != nil {
        log.Fatalf("GetProduct failed: %v", err)
    }
    log.Printf("Product: %+v", resp.Product)
    
    // Create product
    createResp, err := client.CreateProduct(ctx, &pb.CreateProductRequest{
        Name:        "Laptop",
        Description: "High-performance laptop",
        Price:       999.99,
        Stock:       10,
        Category:    pb.ProductCategory_PRODUCT_CATEGORY_ELECTRONICS,
        Tags:        []string{"laptop", "computer", "electronics"},
    })
    if err != nil {
        log.Fatalf("CreateProduct failed: %v", err)
    }
    log.Printf("Created product: %+v", createResp.Product)
    
    // Server streaming
    listReq := &pb.ListProductsRequest{
        Page:     1,
        PageSize: 10,
    }
    
    stream, err := client.ListProducts(ctx, listReq)
    if err != nil {
        log.Fatalf("ListProducts failed: %v", err)
    }
    
    for {
        product, err := stream.Recv()
        if err == io.EOF {
            break
        }
        if err != nil {
            log.Fatalf("Stream error: %v", err)
        }
        log.Printf("Received product: %+v", product)
    }
}
```

### Advanced Client with Interceptors

```go
package main

import (
    "context"
    "log"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
    pb "github.com/example/ecommerce/pb"
)

// Logging interceptor
func loggingInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
    start := time.Now()
    log.Printf("RPC: %s, Request: %+v", info.FullMethod, req)
    
    resp, err := handler(ctx, req)
    
    log.Printf("RPC: %s, Response: %+v, Duration: %v", info.FullMethod, resp, time.Since(start))
    return resp, err
}

// Retry interceptor
func retryInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
    maxRetries := 3
    var lastErr error
    
    for i := 0; i < maxRetries; i++ {
        resp, err := handler(ctx, req)
        if err == nil {
            return resp, nil
        }
        
        lastErr = err
        if i < maxRetries-1 {
            time.Sleep(time.Duration(i+1) * time.Second)
        }
    }
    
    return nil, lastErr
}

func main() {
    // Create connection with interceptors
    conn, err := grpc.Dial("localhost:50051", 
        grpc.WithTransportCredentials(insecure.NewCredentials()),
        grpc.WithUnaryInterceptor(loggingInterceptor),
    )
    if err != nil {
        log.Fatalf("Failed to connect: %v", err)
    }
    defer conn.Close()
    
    // Create client
    client := pb.NewProductServiceClient(conn)
    
    // Use client...
}
```

---

## ❌ Error Handling

### Custom Error Types

```go
package main

import (
    "fmt"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

// Custom error types
type ValidationError struct {
    Field   string
    Message string
}

func (e ValidationError) Error() string {
    return fmt.Sprintf("validation error in field '%s': %s", e.Field, e.Message)
}

type BusinessError struct {
    Code    string
    Message string
}

func (e BusinessError) Error() string {
    return fmt.Sprintf("business error [%s]: %s", e.Code, e.Message)
}

// Convert to gRPC status
func (e ValidationError) ToGRPCStatus() error {
    return status.Error(codes.InvalidArgument, e.Error())
}

func (e BusinessError) ToGRPCStatus() error {
    return status.Error(codes.FailedPrecondition, e.Error())
}

// Error handling in service methods
func (s *ProductServer) GetProduct(ctx context.Context, req *pb.GetProductRequest) (*pb.GetProductResponse, error) {
    if req.Id <= 0 {
        err := ValidationError{Field: "id", Message: "ID must be positive"}
        return nil, err.ToGRPCStatus()
    }
    
    product, exists := s.products[req.Id]
    if !exists {
        err := BusinessError{Code: "PRODUCT_NOT_FOUND", Message: "Product not found"}
        return nil, err.ToGRPCStatus()
    }
    
    return &pb.GetProductResponse{Product: product}, nil
}

// Error handling in client
func handleGRPCError(err error) {
    if err == nil {
        return
    }
    
    st, ok := status.FromError(err)
    if !ok {
        log.Printf("Non-gRPC error: %v", err)
        return
    }
    
    switch st.Code() {
    case codes.NotFound:
        log.Printf("Resource not found: %s", st.Message())
    case codes.InvalidArgument:
        log.Printf("Invalid argument: %s", st.Message())
    case codes.FailedPrecondition:
        log.Printf("Business logic error: %s", st.Message())
    case codes.Internal:
        log.Printf("Internal server error: %s", st.Message())
    default:
        log.Printf("gRPC error [%s]: %s", st.Code(), st.Message())
    }
}
```

---

## 🔐 Authentication & Security

### JWT Authentication

```go
package main

import (
    "context"
    "strings"
    "time"

    "github.com/golang-jwt/jwt/v4"
    "google.golang.org/grpc"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/metadata"
    "google.golang.org/grpc/status"
)

type Claims struct {
    UserID int32  `json:"user_id"`
    Email  string `json:"email"`
    jwt.RegisteredClaims
}

func authInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
    // Skip auth for certain methods
    if info.FullMethod == "/ecommerce.ProductService/GetProduct" {
        return handler(ctx, req)
    }
    
    // Extract token from metadata
    md, ok := metadata.FromIncomingContext(ctx)
    if !ok {
        return nil, status.Error(codes.Unauthenticated, "missing metadata")
    }
    
    authHeader := md.Get("authorization")
    if len(authHeader) == 0 {
        return nil, status.Error(codes.Unauthenticated, "missing authorization header")
    }
    
    token := strings.TrimPrefix(authHeader[0], "Bearer ")
    if token == authHeader[0] {
        return nil, status.Error(codes.Unauthenticated, "invalid authorization header format")
    }
    
    // Validate token
    claims, err := validateToken(token)
    if err != nil {
        return nil, status.Error(codes.Unauthenticated, "invalid token")
    }
    
    // Add user info to context
    ctx = context.WithValue(ctx, "user_id", claims.UserID)
    ctx = context.WithValue(ctx, "user_email", claims.Email)
    
    return handler(ctx, req)
}

func validateToken(tokenString string) (*Claims, error) {
    token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
        return []byte("your-secret-key"), nil
    })
    
    if err != nil {
        return nil, err
    }
    
    if claims, ok := token.Claims.(*Claims); ok && token.Valid {
        return claims, nil
    }
    
    return nil, fmt.Errorf("invalid token")
}

// Client with authentication
func createAuthenticatedClient() pb.ProductServiceClient {
    conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
    if err != nil {
        log.Fatalf("Failed to connect: %v", err)
    }
    
    return pb.NewProductServiceClient(conn)
}

func callWithAuth(client pb.ProductServiceClient, token string) {
    ctx := context.Background()
    md := metadata.Pairs("authorization", "Bearer "+token)
    ctx = metadata.NewOutgoingContext(ctx, md)
    
    resp, err := client.GetProduct(ctx, &pb.GetProductRequest{Id: 1})
    if err != nil {
        log.Fatalf("GetProduct failed: %v", err)
    }
    
    log.Printf("Product: %+v", resp.Product)
}
```

### mTLS (Mutual TLS)

```go
package main

import (
    "crypto/tls"
    "crypto/x509"
    "io/ioutil"
    "log"

    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials"
)

func createTLSCredentials() credentials.TransportCredentials {
    // Load server certificate
    serverCert, err := tls.LoadX509KeyPair("server.crt", "server.key")
    if err != nil {
        log.Fatalf("Failed to load server certificate: %v", err)
    }
    
    // Load CA certificate
    caCert, err := ioutil.ReadFile("ca.crt")
    if err != nil {
        log.Fatalf("Failed to read CA certificate: %v", err)
    }
    
    caCertPool := x509.NewCertPool()
    if !caCertPool.AppendCertsFromPEM(caCert) {
        log.Fatalf("Failed to parse CA certificate")
    }
    
    // Create TLS config
    config := &tls.Config{
        Certificates: []tls.Certificate{serverCert},
        ClientCAs:    caCertPool,
        ClientAuth:   tls.RequireAndVerifyClientCert,
    }
    
    return credentials.NewTLS(config)
}

func main() {
    // Create server with TLS
    lis, err := net.Listen("tcp", ":50051")
    if err != nil {
        log.Fatalf("Failed to listen: %v", err)
    }
    
    creds := createTLSCredentials()
    s := grpc.NewServer(grpc.Creds(creds))
    
    // Register services
    pb.RegisterProductServiceServer(s, &ProductServer{})
    
    log.Println("Server starting with mTLS on :50051")
    if err := s.Serve(lis); err != nil {
        log.Fatalf("Failed to serve: %v", err)
    }
}
```

---

## ⚖️ Load Balancing

### Client-side Load Balancing

```go
package main

import (
    "context"
    "log"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/balancer/roundrobin"
    "google.golang.org/grpc/resolver"
    pb "github.com/example/ecommerce/pb"
)

func main() {
    // Register custom resolver
    resolver.Register(&customResolver{})
    
    // Create connection with load balancing
    conn, err := grpc.Dial("custom:///product-service", 
        grpc.WithInsecure(),
        grpc.WithBalancerName(roundrobin.Name),
    )
    if err != nil {
        log.Fatalf("Failed to connect: %v", err)
    }
    defer conn.Close()
    
    client := pb.NewProductServiceClient(conn)
    
    // Make requests (will be load balanced)
    for i := 0; i < 10; i++ {
        resp, err := client.GetProduct(context.Background(), &pb.GetProductRequest{Id: 1})
        if err != nil {
            log.Printf("Request %d failed: %v", i, err)
        } else {
            log.Printf("Request %d succeeded: %+v", i, resp.Product)
        }
        time.Sleep(time.Second)
    }
}

type customResolver struct{}

func (r *customResolver) ResolveNow(resolver.ResolveNowOptions) {}
func (r *customResolver) Close() {}

func (r *customResolver) Build(target resolver.Target, cc resolver.ClientConn, opts resolver.BuildOptions) (resolver.Resolver, error) {
    // Add multiple addresses
    addresses := []resolver.Address{
        {Addr: "localhost:50051"},
        {Addr: "localhost:50052"},
        {Addr: "localhost:50053"},
    }
    
    cc.UpdateState(resolver.State{Addresses: addresses})
    return r, nil
}
```

### Health Checking

```go
package main

import (
    "context"
    "log"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/health"
    "google.golang.org/grpc/health/grpc_health_v1"
    "google.golang.org/grpc/reflection"
    pb "github.com/example/ecommerce/pb"
)

func main() {
    lis, err := net.Listen("tcp", ":50051")
    if err != nil {
        log.Fatalf("Failed to listen: %v", err)
    }
    
    s := grpc.NewServer()
    
    // Register services
    pb.RegisterProductServiceServer(s, &ProductServer{})
    
    // Register health service
    healthServer := health.NewServer()
    grpc_health_v1.RegisterHealthServer(s, healthServer)
    
    // Set service status
    healthServer.SetServingStatus("ecommerce.ProductService", grpc_health_v1.HealthCheckResponse_SERVING)
    
    // Register reflection service
    reflection.Register(s)
    
    log.Println("Server starting on :50051")
    if err := s.Serve(lis); err != nil {
        log.Fatalf("Failed to serve: %v", err)
    }
}

// Health check client
func checkHealth() {
    conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
    if err != nil {
        log.Fatalf("Failed to connect: %v", err)
    }
    defer conn.Close()
    
    client := grpc_health_v1.NewHealthClient(conn)
    
    resp, err := client.Check(context.Background(), &grpc_health_v1.HealthCheckRequest{
        Service: "ecommerce.ProductService",
    })
    if err != nil {
        log.Fatalf("Health check failed: %v", err)
    }
    
    log.Printf("Health status: %s", resp.Status)
}
```

---

## 📊 Monitoring & Observability

### Metrics and Tracing

```go
package main

import (
    "context"
    "time"

    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/trace"
    "google.golang.org/grpc"
    "google.golang.org/grpc/codes"
    pb "github.com/example/ecommerce/pb"
)

var (
    // Prometheus metrics
    requestDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "grpc_request_duration_seconds",
            Help: "Duration of gRPC requests",
        },
        []string{"method", "status"},
    )
    
    requestTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "grpc_requests_total",
            Help: "Total number of gRPC requests",
        },
        []string{"method", "status"},
    )
)

// Metrics interceptor
func metricsInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
    start := time.Now()
    
    resp, err := handler(ctx, req)
    
    duration := time.Since(start).Seconds()
    status := "success"
    if err != nil {
        status = "error"
    }
    
    requestDuration.WithLabelValues(info.FullMethod, status).Observe(duration)
    requestTotal.WithLabelValues(info.FullMethod, status).Inc()
    
    return resp, err
}

// Tracing interceptor
func tracingInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
    tracer := otel.Tracer("product-service")
    ctx, span := tracer.Start(ctx, info.FullMethod)
    defer span.End()
    
    // Add attributes
    span.SetAttributes(
        attribute.String("grpc.method", info.FullMethod),
        attribute.String("grpc.service", "ProductService"),
    )
    
    resp, err := handler(ctx, req)
    
    if err != nil {
        span.RecordError(err)
        span.SetStatus(codes.Error, err.Error())
    }
    
    return resp, err
}

// Logging interceptor
func loggingInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
    start := time.Now()
    
    log.Printf("gRPC call: %s", info.FullMethod)
    
    resp, err := handler(ctx, req)
    
    duration := time.Since(start)
    if err != nil {
        log.Printf("gRPC call failed: %s, duration: %v, error: %v", info.FullMethod, duration, err)
    } else {
        log.Printf("gRPC call succeeded: %s, duration: %v", info.FullMethod, duration)
    }
    
    return resp, err
}

func main() {
    // Create server with interceptors
    s := grpc.NewServer(
        grpc.UnaryInterceptor(grpc.ChainUnaryInterceptor(
            loggingInterceptor,
            metricsInterceptor,
            tracingInterceptor,
        )),
    )
    
    // Register services
    pb.RegisterProductServiceServer(s, &ProductServer{})
    
    // Start server
    lis, err := net.Listen("tcp", ":50051")
    if err != nil {
        log.Fatalf("Failed to listen: %v", err)
    }
    
    log.Println("Server starting on :50051")
    if err := s.Serve(lis); err != nil {
        log.Fatalf("Failed to serve: %v", err)
    }
}
```

---

## 🏆 Best Practices

### 1. Service Design

```go
// Good: Clear service boundaries
service ProductService {
  rpc GetProduct(GetProductRequest) returns (GetProductResponse);
  rpc ListProducts(ListProductsRequest) returns (stream Product);
  rpc CreateProduct(CreateProductRequest) returns (CreateProductResponse);
  rpc UpdateProduct(UpdateProductRequest) returns (UpdateProductResponse);
  rpc DeleteProduct(DeleteProductRequest) returns (google.protobuf.Empty);
}

// Bad: Mixed responsibilities
service ProductService {
  rpc GetProduct(GetProductRequest) returns (GetProductResponse);
  rpc ProcessPayment(PaymentRequest) returns (PaymentResponse); // Wrong service
  rpc SendEmail(EmailRequest) returns (EmailResponse); // Wrong service
}
```

### 2. Message Design

```go
// Good: Specific request/response messages
message GetProductRequest {
  int32 id = 1;
}

message GetProductResponse {
  Product product = 1;
}

// Bad: Generic messages
message GenericRequest {
  string data = 1;
}

message GenericResponse {
  string data = 1;
}
```

### 3. Error Handling

```go
// Good: Specific error codes and messages
func (s *ProductServer) GetProduct(ctx context.Context, req *pb.GetProductRequest) (*pb.GetProductResponse, error) {
    if req.Id <= 0 {
        return nil, status.Error(codes.InvalidArgument, "Product ID must be positive")
    }
    
    product, exists := s.products[req.Id]
    if !exists {
        return nil, status.Error(codes.NotFound, "Product not found")
    }
    
    return &pb.GetProductResponse{Product: product}, nil
}

// Bad: Generic error handling
func (s *ProductServer) GetProduct(ctx context.Context, req *pb.GetProductRequest) (*pb.GetProductResponse, error) {
    product, exists := s.products[req.Id]
    if !exists {
        return nil, fmt.Errorf("error") // Not helpful
    }
    
    return &pb.GetProductResponse{Product: product}, nil
}
```

### 4. Performance Optimization

```go
// Good: Use streaming for large datasets
func (s *ProductServer) ListProducts(req *pb.ListProductsRequest, stream pb.ProductService_ListProductsServer) error {
    for _, product := range s.products {
        if err := stream.Send(product); err != nil {
            return err
        }
    }
    return nil
}

// Bad: Loading all data into memory
func (s *ProductServer) ListProducts(ctx context.Context, req *pb.ListProductsRequest) (*pb.ListProductsResponse, error) {
    var products []*pb.Product
    for _, product := range s.products {
        products = append(products, product)
    }
    
    return &pb.ListProductsResponse{Products: products}, nil
}
```

---

## 🌟 Real-world Examples

### E-commerce Microservices

```go
// Product Service
service ProductService {
  rpc GetProduct(GetProductRequest) returns (GetProductResponse);
  rpc ListProducts(ListProductsRequest) returns (stream Product);
  rpc CreateProduct(CreateProductRequest) returns (CreateProductResponse);
  rpc UpdateProduct(UpdateProductRequest) returns (UpdateProductResponse);
  rpc DeleteProduct(DeleteProductRequest) returns (google.protobuf.Empty);
  rpc SearchProducts(SearchProductsRequest) returns (stream Product);
}

// Order Service
service OrderService {
  rpc CreateOrder(CreateOrderRequest) returns (CreateOrderResponse);
  rpc GetOrder(GetOrderRequest) returns (GetOrderResponse);
  rpc UpdateOrderStatus(UpdateOrderStatusRequest) returns (UpdateOrderStatusResponse);
  rpc ListOrders(ListOrdersRequest) returns (stream Order);
  rpc StreamOrderUpdates(GetOrderRequest) returns (stream OrderUpdate);
}

// Payment Service
service PaymentService {
  rpc ProcessPayment(PaymentRequest) returns (PaymentResponse);
  rpc RefundPayment(RefundRequest) returns (RefundResponse);
  rpc GetPaymentHistory(GetPaymentHistoryRequest) returns (stream Payment);
}

// Notification Service
service NotificationService {
  rpc SendNotification(NotificationRequest) returns (NotificationResponse);
  rpc StreamNotifications(StreamNotificationsRequest) returns (stream Notification);
}
```

### Chat Application

```go
service ChatService {
  rpc JoinRoom(JoinRoomRequest) returns (stream ChatMessage);
  rpc SendMessage(stream ChatMessage) returns (google.protobuf.Empty);
  rpc GetRoomHistory(GetRoomHistoryRequest) returns (stream ChatMessage);
  rpc GetOnlineUsers(GetOnlineUsersRequest) returns (stream User);
}
```

### Real-time Analytics

```go
service AnalyticsService {
  rpc TrackEvent(EventRequest) returns (EventResponse);
  rpc GetMetrics(GetMetricsRequest) returns (GetMetricsResponse);
  rpc StreamMetrics(StreamMetricsRequest) returns (stream Metric);
  rpc GetDashboard(DashboardRequest) returns (DashboardResponse);
}
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
# Install protobuf compiler
brew install protobuf

# Install Go plugins
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
```

### 2. Generate Code

```bash
# Generate Go code from proto files
protoc --go_out=. --go_opt=paths=source_relative \
    --go-grpc_out=. --go-grpc_opt=paths=source_relative \
    product.proto
```

### 3. Run Server

```bash
go run server/main.go
```

### 4. Run Client

```bash
go run client/main.go
```

---

**🎉 You now have a comprehensive understanding of gRPC! Use this knowledge to build high-performance microservices and ace your Razorpay interviews! 🚀**
