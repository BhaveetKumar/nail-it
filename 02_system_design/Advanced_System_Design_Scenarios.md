# 🚀 **Advanced System Design Scenarios**

## 📊 **Complex Real-World System Design Problems with Solutions**

---

## 🎯 **1. Design a Real-Time Chat System (WhatsApp/Slack)**

### **Requirements Analysis**

#### **Functional Requirements**
- Send/receive messages in real-time
- Support group chats
- Message history and search
- Online/offline status
- File sharing (images, documents)
- Push notifications

#### **Non-Functional Requirements**
- **Scale**: 1 billion users, 50 billion messages/day
- **Latency**: < 100ms for message delivery
- **Availability**: 99.9%
- **Consistency**: Eventually consistent

### **High-Level Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mobile App    │    │   Web Client    │    │  Desktop App    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      Load Balancer        │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      API Gateway          │
                    └─────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼─────────┐  ┌─────────▼─────────┐  ┌─────────▼─────────┐
│  Message Service  │  │   User Service    │  │  Notification     │
│                   │  │                   │  │     Service       │
└─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │     Message Queue         │
                    │      (Apache Kafka)       │
                    └─────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼─────────┐  ┌─────────▼─────────┐  ┌─────────▼─────────┐
│   WebSocket       │  │   Message Store   │  │   File Storage    │
│   Connection      │  │    (Cassandra)    │  │      (S3)         │
│    Manager        │  │                   │  │                   │
└───────────────────┘  └───────────────────┘  └───────────────────┘
```

### **Key Components**

#### **1. Message Service**
- **Real-time messaging** with WebSocket connections
- **Message persistence** in Cassandra
- **Message queuing** for reliable delivery
- **Push notifications** for offline users

#### **2. WebSocket Manager**
- **Connection management** for real-time communication
- **Message broadcasting** to multiple clients
- **Connection pooling** for scalability
- **Heartbeat mechanism** for connection health

#### **3. Message Store (Cassandra)**
- **Distributed storage** for message history
- **Time-series data** with clustering by timestamp
- **Sharding by user ID** for scalability
- **Eventual consistency** for performance

### **Database Schema**

```sql
-- Messages table
CREATE TABLE messages (
    id UUID PRIMARY KEY,
    sender_id UUID,
    receiver_id UUID,
    group_id UUID,
    content TEXT,
    type TEXT,
    timestamp TIMESTAMP,
    status TEXT
) WITH CLUSTERING ORDER BY (timestamp DESC);

-- Group messages table
CREATE TABLE group_messages (
    group_id UUID,
    timestamp TIMESTAMP,
    id UUID,
    sender_id UUID,
    content TEXT,
    type TEXT,
    status TEXT,
    PRIMARY KEY (group_id, timestamp, id)
) WITH CLUSTERING ORDER BY (timestamp DESC);
```

---

## 🎯 **2. Design a Video Streaming Platform (Netflix/YouTube)**

### **Requirements Analysis**

#### **Functional Requirements**
- Upload and process videos
- Stream videos to users
- Video recommendations
- User profiles and watch history
- Comments and ratings
- Live streaming support

#### **Non-Functional Requirements**
- **Scale**: 1 billion users, 1 billion hours watched daily
- **Latency**: < 2 seconds for video start
- **Availability**: 99.9%
- **Storage**: Petabytes of video data

### **High-Level Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mobile App    │    │   Web Client    │    │  Smart TV App   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      CDN (CloudFront)     │
                    └─────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼─────────┐  ┌─────────▼─────────┐  ┌─────────▼─────────┐
│  Video Upload     │  │  Video Streaming  │  │  Recommendation   │
│     Service       │  │     Service       │  │     Service       │
└─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │     Video Processing      │
                    │        Pipeline           │
                    └─────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼─────────┐  ┌─────────▼─────────┐  ┌─────────▼─────────┐
│   Video Storage   │  │   Metadata DB     │  │   Analytics DB    │
│      (S3)         │  │   (DynamoDB)      │  │   (Redshift)      │
└───────────────────┘  └───────────────────┘  └───────────────────┘
```

### **Key Components**

#### **1. Video Upload Service**
- **File upload** with chunked transfer
- **Metadata extraction** (duration, resolution, etc.)
- **Thumbnail generation** for preview
- **Processing queue** for video encoding

#### **2. Video Processing Pipeline**
- **Multiple quality encoding** (240p, 360p, 480p, 720p, 1080p)
- **Adaptive bitrate streaming** (HLS/DASH)
- **Thumbnail generation** at different timestamps
- **Subtitle processing** and synchronization

#### **3. Video Streaming Service**
- **CDN integration** for global delivery
- **Adaptive quality selection** based on bandwidth
- **Signed URLs** for secure access
- **Analytics tracking** for user behavior

---

## 🎯 **3. Design a Social Media Feed (Facebook/Twitter)**

### **Requirements Analysis**

#### **Functional Requirements**
- Post updates (text, images, videos)
- Follow/unfollow users
- Like and comment on posts
- Real-time feed updates
- Search posts and users
- Trending topics

#### **Non-Functional Requirements**
- **Scale**: 2 billion users, 500 million posts/day
- **Latency**: < 200ms for feed generation
- **Availability**: 99.9%
- **Consistency**: Eventually consistent

### **High-Level Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mobile App    │    │   Web Client    │    │  Desktop App    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      Load Balancer        │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      API Gateway          │
                    └─────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼─────────┐  ┌─────────▼─────────┐  ┌─────────▼─────────┐
│   Feed Service    │  │   Post Service    │  │   User Service    │
└─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │     Message Queue         │
                    │      (Apache Kafka)       │
                    └─────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼─────────┐  ┌─────────▼─────────┐  ┌─────────▼─────────┐
│   Feed Cache      │  │   Post Store      │  │   User Store      │
│    (Redis)        │  │   (Cassandra)     │  │   (MySQL)         │
└───────────────────┘  └───────────────────┘  └───────────────────┘
```

### **Key Components**

#### **1. Feed Service**
- **Feed generation** from followed users
- **Caching strategy** for performance
- **Real-time updates** via message queue
- **Personalization** based on user behavior

#### **2. Post Service**
- **Post creation** and management
- **Like and comment** functionality
- **Content moderation** and filtering
- **Search and discovery** features

#### **3. User Service**
- **User profiles** and authentication
- **Follow/unfollow** relationships
- **Privacy settings** and controls
- **User recommendations** based on interests

---

## 🎯 **4. Design a Distributed Cache System (Redis Cluster)**

### **Requirements Analysis**

#### **Functional Requirements**
- Store and retrieve key-value pairs
- Support different data types (string, hash, list, set)
- Expiration and TTL support
- Atomic operations
- Pub/Sub messaging

#### **Non-Functional Requirements**
- **Scale**: 1 million operations/second
- **Latency**: < 1ms for cache hits
- **Availability**: 99.99%
- **Consistency**: Eventually consistent

### **High-Level Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │    │   Client App    │    │   Client App    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    Load Balancer          │
                    └─────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼─────────┐  ┌─────────▼─────────┐  ┌─────────▼─────────┐
│   Redis Node 1    │  │   Redis Node 2    │  │   Redis Node 3    │
│   (Master)        │  │   (Master)        │  │   (Master)        │
└─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    Consistent Hashing     │
                    │        Algorithm          │
                    └───────────────────────────┘
```

### **Key Components**

#### **1. Consistent Hashing**
- **Virtual nodes** for load balancing
- **Hash ring** for key distribution
- **Node addition/removal** with minimal data movement
- **Replication** for fault tolerance

#### **2. Redis Cluster**
- **Master-slave replication** for high availability
- **Automatic failover** when master fails
- **Data sharding** across multiple nodes
- **Gossip protocol** for cluster communication

#### **3. Client Library**
- **Connection pooling** for performance
- **Automatic failover** handling
- **Load balancing** across cluster nodes
- **Circuit breaker** for fault tolerance

---

## 🎯 **5. Design a Search Engine (Google/Elasticsearch)**

### **Requirements Analysis**

#### **Functional Requirements**
- Index and search documents
- Full-text search with relevance scoring
- Faceted search and filtering
- Autocomplete and suggestions
- Real-time indexing

#### **Non-Functional Requirements**
- **Scale**: 1 billion documents, 10,000 queries/second
- **Latency**: < 100ms for search results
- **Availability**: 99.9%
- **Relevance**: High-quality search results

### **High-Level Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │   Mobile App    │    │   API Client    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      Load Balancer        │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │     Search Service        │
                    └─────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼─────────┐  ┌─────────▼─────────┐  ┌─────────▼─────────┐
│   Index Service   │  │   Query Service   │  │  Suggestion       │
│                   │  │                   │  │   Service         │
└─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    Elasticsearch          │
                    │       Cluster             │
                    └─────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼─────────┐  ┌─────────▼─────────┐  ┌─────────▼─────────┐
│   Document Store  │  │   Index Store     │  │   Analytics       │
│      (S3)         │  │   (Elasticsearch) │  │   (Kafka)         │
└───────────────────┘  └───────────────────┘  └───────────────────┘
```

### **Key Components**

#### **1. Index Service**
- **Document ingestion** and preprocessing
- **Text analysis** and tokenization
- **Index creation** and management
- **Real-time updates** for new documents

#### **2. Query Service**
- **Query parsing** and optimization
- **Relevance scoring** algorithms
- **Faceted search** and filtering
- **Result ranking** and pagination

#### **3. Suggestion Service**
- **Autocomplete** functionality
- **Query suggestions** based on history
- **Spell correction** and fuzzy matching
- **Trending queries** and popular searches

---

## 🎯 **Key Takeaways**

### **1. Real-Time Chat System**
- **WebSocket connections** for real-time communication
- **Message queuing** for reliable delivery
- **Database sharding** for scalability
- **Push notifications** for offline users

### **2. Video Streaming Platform**
- **CDN** for global content delivery
- **Video processing pipeline** for multiple qualities
- **Storage optimization** for large files
- **Analytics** for user behavior tracking

### **3. Social Media Feed**
- **Feed caching** for performance
- **Fan-out pattern** for real-time updates
- **Event-driven architecture** for decoupling
- **Search optimization** for content discovery

### **4. Distributed Cache System**
- **Consistent hashing** for load distribution
- **Master-slave replication** for availability
- **Automatic failover** for fault tolerance
- **Connection pooling** for performance

### **5. Search Engine**
- **Inverted index** for fast text search
- **Relevance scoring** for quality results
- **Faceted search** for advanced filtering
- **Real-time indexing** for fresh content

---

**🎉 This comprehensive guide provides advanced system design scenarios with complete solutions for interview success! 🚀**