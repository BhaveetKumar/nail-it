# Composite Pattern Implementation

This microservice demonstrates the **Composite Pattern** implementation in Go, providing a way to compose objects into tree structures to represent part-whole hierarchies, allowing clients to treat individual objects and compositions uniformly.

## Overview

The Composite Pattern allows you to compose objects into tree structures to represent part-whole hierarchies. In this implementation:

- **Component**: Common interface for both leaf and composite objects
- **Leaf**: Individual objects that don't have children (files, menu items)
- **Composite**: Objects that can contain other components (folders, menus)
- **Client**: Uses components uniformly without knowing if they're leaf or composite

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client        │    │   Composite      │    │   Components    │
│                 │    │   Service        │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Create      │◄┼────┼─┤ CreateFile   │◄┼────┼─┤ FileSystem  │ │
│ │ FileSystem  │ │    │ │ System       │ │    │ │ Component   │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Create      │◄┼────┼─┤ CreateMenu   │◄┼────┼─┤ MenuSystem  │ │
│ │ MenuSystem  │ │    │ │ System       │ │    │ │ Component   │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Add         │◄┼────┼─┤ AddComponent │◄┼────┼─┤ Folder      │ │
│ │ Component   │ │    │ │              │ │    │ │ Component   │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Execute     │◄┼────┼─┤ Execute      │◄┼────┼─┤ File        │ │
│ │ Component   │ │    │ │ Component    │ │    │ │ Component   │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Get Tree    │◄┼────┼─┤ GetComponent │◄┼────┼─┤ Menu        │ │
│ │ Structure   │ │    │ │ Tree         │ │    │ │ Component   │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │                  │    │                 │
│                 │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│                 │    │ │ Get          │ │    │ │ Base        │ │
│                 │    │ │ Statistics   │ │    │ │ Component   │ │
│                 │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │                  │    │                 │
│                 │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│                 │    │ │ Optimize     │ │    │ │ Composite   │ │
│                 │    │ │ Component    │ │    │ │ Component   │ │
│                 │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    └──────────────────┘    └─────────────────┘
└─────────────────┘
```

## Features

### Core Component Types
- **FileSystemComponent**: Root component for file system operations
- **MenuSystemComponent**: Root component for menu system operations
- **FolderComponent**: Composite component for directories
- **FileComponent**: Leaf component for files
- **MenuComponent**: Composite component for menu items

### Component Operations
- **Add/Remove**: Add or remove child components
- **Execute**: Execute component operations
- **Traverse**: Tree traversal with visitors
- **Statistics**: Component tree statistics
- **Optimization**: Tree structure optimization

### Service Layer
- **High-Level Operations**: Simplified API for component operations
- **Tree Management**: Complete tree structure management
- **Caching**: Multi-level caching for optimal performance
- **Database Integration**: Persistent storage of components

## API Endpoints

### File System Operations
```bash
# Create file system
POST /api/v1/filesystems
Content-Type: application/json

{
  "id": "fs_001",
  "name": "My File System",
  "root_path": "/home/user"
}

# Get file system
GET /api/v1/filesystems/{id}

# Execute file system
POST /api/v1/filesystems/{id}/execute

# Get file system tree
GET /api/v1/filesystems/{id}/tree

# Get file system statistics
GET /api/v1/filesystems/{id}/statistics

# Optimize file system
POST /api/v1/filesystems/{id}/optimize
```

### Menu System Operations
```bash
# Create menu system
POST /api/v1/menus
Content-Type: application/json

{
  "id": "menu_001",
  "name": "Main Menu",
  "base_url": "https://example.com"
}

# Get menu system
GET /api/v1/menus/{id}

# Execute menu system
POST /api/v1/menus/{id}/execute
```

### Component Management
```bash
# Add component
POST /api/v1/components/add
Content-Type: application/json

{
  "parent_id": "fs_001",
  "child_id": "folder_001",
  "child_name": "Documents",
  "component_type": "folder",
  "path": "/home/user/Documents"
}

# Remove component
DELETE /api/v1/components/remove
Content-Type: application/json

{
  "parent_id": "fs_001",
  "child_id": "folder_001"
}

# Get component
GET /api/v1/components/{id}

# Execute component
POST /api/v1/components/{id}/execute
```

### Health Check
```bash
# Health check
GET /health
```

### WebSocket
```bash
# Connect for real-time updates
WS /ws
```

## Configuration

The service uses YAML configuration with the following key sections:

- **Server**: Port, timeouts, CORS settings
- **Database**: MySQL, MongoDB, Redis configurations
- **Kafka**: Message queue settings
- **Composite**: Composite-specific settings
- **Cache**: Caching configuration
- **Message Queue**: Queue settings
- **WebSocket**: Real-time communication settings
- **Security**: Authentication and authorization
- **Monitoring**: Metrics and health checks
- **Business Logic**: Feature flags and optimizations

## Dependencies

```go
require (
    github.com/gin-gonic/gin v1.9.1
    github.com/gorilla/websocket v1.5.0
    github.com/go-redis/redis/v8 v8.11.5
    github.com/Shopify/sarama v1.38.1
    go.mongodb.org/mongo-driver v1.12.1
    gorm.io/driver/mysql v1.5.2
    gorm.io/gorm v1.25.5
    go.uber.org/zap v1.25.0
    github.com/patrickmn/go-cache v2.1.0+incompatible
    github.com/prometheus/client_golang v1.17.0
    github.com/sirupsen/logrus v1.9.3
    github.com/opentracing/opentracing-go v1.2.0
    github.com/uber/jaeger-client-go v2.30.0+incompatible
)
```

## Running the Service

1. **Start dependencies**:
   ```bash
   # Start MySQL
   docker run -d --name mysql -e MYSQL_ROOT_PASSWORD=password -e MYSQL_DATABASE=composite_db -p 3306:3306 mysql:8.0
   
   # Start MongoDB
   docker run -d --name mongodb -p 27017:27017 mongo:6.0
   
   # Start Redis
   docker run -d --name redis -p 6379:6379 redis:7.0
   
   # Start Kafka
   docker run -d --name kafka -p 9092:9092 apache/kafka:3.4.0
   ```

2. **Run the service**:
   ```bash
   go mod tidy
   go run main.go
   ```

3. **Test the service**:
   ```bash
   # Health check
   curl http://localhost:8080/health
   
   # Create file system
   curl -X POST http://localhost:8080/api/v1/filesystems \
     -H "Content-Type: application/json" \
     -d '{"id":"fs_001","name":"My File System","root_path":"/home/user"}'
   
   # Add folder to file system
   curl -X POST http://localhost:8080/api/v1/components/add \
     -H "Content-Type: application/json" \
     -d '{"parent_id":"fs_001","child_id":"folder_001","child_name":"Documents","component_type":"folder","path":"/home/user/Documents"}'
   ```

## Design Patterns Used

### Composite Pattern
- **Component Interface**: Common interface for all components
- **Leaf Components**: Individual objects (files, menu items)
- **Composite Components**: Objects that can contain children (folders, menus)
- **Uniform Treatment**: Clients treat leaf and composite objects uniformly

### Additional Patterns
- **Factory Pattern**: Component creation and management
- **Visitor Pattern**: Tree traversal and operations
- **Iterator Pattern**: Component iteration
- **Service Layer Pattern**: High-level business operations
- **Observer Pattern**: WebSocket real-time updates

## Benefits

1. **Uniformity**: Treat leaf and composite objects uniformly
2. **Flexibility**: Easy to add new component types
3. **Scalability**: Handle complex tree structures efficiently
4. **Maintainability**: Clear separation of concerns
5. **Reusability**: Components can be reused in different contexts
6. **Tree Operations**: Built-in support for tree traversal and statistics
7. **Performance**: Optimized tree operations and caching

## Use Cases

- **File Systems**: Directory and file management
- **Menu Systems**: Hierarchical navigation menus
- **UI Components**: Nested UI component trees
- **Document Structures**: Hierarchical document organization
- **Organization Charts**: Employee hierarchy management
- **Product Catalogs**: Category and product organization
- **Configuration Management**: Nested configuration structures

## Tree Structure Examples

### File System Structure
```
FileSystem (fs_001)
├── Documents (folder_001)
│   ├── Work (folder_002)
│   │   ├── project1.txt (file_001)
│   │   └── project2.txt (file_002)
│   └── Personal (folder_003)
│       └── notes.txt (file_003)
├── Pictures (folder_004)
│   └── vacation.jpg (file_004)
└── Downloads (folder_005)
    └── software.zip (file_005)
```

### Menu System Structure
```
MenuSystem (menu_001)
├── Home (menu_002)
├── Products (menu_003)
│   ├── Electronics (menu_004)
│   │   ├── Phones (menu_005)
│   │   └── Laptops (menu_006)
│   └── Clothing (menu_007)
│       ├── Men (menu_008)
│       └── Women (menu_009)
├── About (menu_010)
└── Contact (menu_011)
```

## Performance Characteristics

### Tree Operations
- **Add Component**: O(1) for direct children, O(n) for deep trees
- **Remove Component**: O(1) for direct children, O(n) for deep trees
- **Tree Traversal**: O(n) where n is the total number of nodes
- **Statistics Calculation**: O(n) where n is the total number of nodes

### Memory Usage
- **Component Storage**: O(n) where n is the total number of components
- **Tree Structure**: O(n) where n is the total number of nodes
- **Cache Overhead**: O(k) where k is the cache size

### Scalability
- **Concurrent Access**: Thread-safe component operations
- **Memory Management**: Automatic cleanup and garbage collection
- **Tree Optimization**: Built-in tree structure optimization

## Monitoring

- **Metrics**: Component operations, tree statistics, performance metrics
- **Health Checks**: Service health and component tree health
- **Logging**: Structured logging with component context
- **Statistics**: Detailed component and tree statistics
- **WebSocket**: Real-time monitoring dashboards

## Security Features

- **Input Validation**: Validate all component data
- **Access Control**: Permission-based component access
- **Audit Logging**: Track component operations
- **Rate Limiting**: Prevent abuse of component operations
- **Data Encryption**: Encrypt sensitive component data

## Performance Features

- **Multi-Level Caching**: Memory + Redis caching
- **Connection Pooling**: Efficient database connections
- **Async Processing**: Non-blocking operations
- **Tree Optimization**: Automatic tree structure optimization
- **Memory Monitoring**: Real-time memory usage tracking
- **Batch Operations**: Efficient bulk operations

## Best Practices

1. **Design for Uniformity**: Ensure leaf and composite objects have consistent interfaces
2. **Limit Tree Depth**: Set reasonable limits on tree depth and children count
3. **Use Appropriate Types**: Choose the right component type for your use case
4. **Implement Validation**: Validate component data and tree structure
5. **Monitor Performance**: Track tree operations and memory usage
6. **Optimize Regularly**: Use built-in optimization features
7. **Handle Errors Gracefully**: Implement proper error handling for tree operations

This implementation demonstrates how the Composite Pattern can be used to create flexible, maintainable systems for managing hierarchical structures, making it ideal for file systems, menu systems, and other tree-like data organizations.
