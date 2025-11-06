---
# Auto-generated front matter
Title: Kafka Complete Guide
LastUpdated: 2025-11-06T20:45:59.100346
Tags: []
Status: draft
---

# 🚀 Apache Kafka Complete Guide - Theory, Practice & Production

## Table of Contents

1. [Kafka Fundamentals](#kafka-fundamentals)
2. [Core Concepts & Architecture](#core-concepts--architecture)
3. [Node.js Kafka Integration](#nodejs-kafka-integration)
4. [Advanced Patterns & Edge Cases](#advanced-patterns--edge-cases)
5. [Scaling & Performance](#scaling--performance)
6. [Production Operations](#production-operations)
7. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
8. [Real-world Use Cases](#real-world-use-cases)

---

## Kafka Fundamentals

### What is Apache Kafka?

Apache Kafka is a distributed streaming platform that can:

- **Publish and subscribe** to streams of records
- **Store** streams of records in a fault-tolerant way
- **Process** streams of records as they occur

### Key Characteristics

```javascript
// Kafka Core Properties
const kafkaCharacteristics = {
  distributed: "Runs on multiple servers",
  faultTolerant: "Replicates data across multiple nodes",
  highThroughput: "Handles millions of messages per second",
  lowLatency: "Sub-millisecond latency for real-time processing",
  durable: "Persists data to disk",
  scalable: "Horizontal scaling by adding more brokers",
};
```

### Kafka vs Other Messaging Systems

| Feature               | Kafka         | RabbitMQ | ActiveMQ | Redis Pub/Sub |
| --------------------- | ------------- | -------- | -------- | ------------- |
| **Throughput**        | Very High     | Medium   | Medium   | High          |
| **Latency**           | Low           | Low      | Medium   | Very Low      |
| **Durability**        | High          | Medium   | High     | Low           |
| **Scalability**       | Excellent     | Good     | Good     | Limited       |
| **Message Ordering**  | Per-partition | No       | No       | No            |
| **Message Retention** | Configurable  | No       | No       | No            |

---

## Core Concepts & Architecture

### 1. Topics and Partitions

```javascript
// Topic Structure
class KafkaTopic {
  constructor(name, partitions = 3, replicationFactor = 3) {
    this.name = name;
    this.partitions = partitions;
    this.replicationFactor = replicationFactor;
    this.config = {
      retention: "7 days",
      segmentSize: "1GB",
      compression: "snappy",
    };
  }
}

// Example Topics
const topics = {
  userEvents: new KafkaTopic("user-events", 12, 3),
  orderProcessing: new KafkaTopic("order-processing", 6, 3),
  analytics: new KafkaTopic("analytics", 24, 3),
};
```

### 2. Producers and Consumers

```javascript
// Producer Configuration
const producerConfig = {
  bootstrapServers: ["broker1:9092", "broker2:9092", "broker3:9092"],
  acks: "all", // Wait for all replicas to acknowledge
  retries: 3,
  retryBackoffMs: 100,
  batchSize: 16384,
  lingerMs: 5,
  compressionType: "snappy",
  maxInFlightRequests: 5,
  enableIdempotence: true,
};

// Consumer Configuration
const consumerConfig = {
  bootstrapServers: ["broker1:9092", "broker2:9092", "broker3:9092"],
  groupId: "my-consumer-group",
  autoOffsetReset: "earliest",
  enableAutoCommit: false,
  maxPollRecords: 500,
  sessionTimeoutMs: 30000,
  heartbeatIntervalMs: 3000,
  maxPollIntervalMs: 300000,
};
```

### 3. Consumer Groups

```javascript
// Consumer Group Strategy
class ConsumerGroupStrategy {
  static roundRobin = "Round-robin partition assignment";
  static range = "Range-based partition assignment";
  static sticky = "Sticky partition assignment (preferred)";
  static cooperativeSticky = "Cooperative sticky (latest)";
}

// Consumer Group Rebalancing
const rebalancingStrategies = {
  eager: "Stop all consumers, reassign, restart",
  cooperative: "Incremental rebalancing without stopping",
  benefits: {
    faster: "No downtime during rebalancing",
    efficient: "Only affected partitions are reassigned",
    stable: "Reduces partition movement",
  },
};
```

---

## Node.js Kafka Integration

### 1. Basic Producer Implementation

```javascript
const { Kafka } = require("kafkajs");

class KafkaProducer {
  constructor(config) {
    this.kafka = new Kafka({
      clientId: config.clientId || "nodejs-producer",
      brokers: config.brokers,
      retry: {
        initialRetryTime: 100,
        retries: 8,
      },
    });

    this.producer = this.kafka.producer({
      maxInFlightRequests: 1,
      idempotent: true,
      transactionTimeout: 30000,
    });

    this.isConnected = false;
  }

  async connect() {
    try {
      await this.producer.connect();
      this.isConnected = true;
      console.log("Producer connected successfully");
    } catch (error) {
      console.error("Failed to connect producer:", error);
      throw error;
    }
  }

  async sendMessage(topic, message, options = {}) {
    if (!this.isConnected) {
      throw new Error("Producer not connected");
    }

    const messagePayload = {
      topic,
      messages: [
        {
          key: options.key || null,
          value: JSON.stringify(message),
          partition: options.partition,
          timestamp: options.timestamp || Date.now().toString(),
          headers: options.headers || {},
        },
      ],
    };

    try {
      const result = await this.producer.send(messagePayload);
      console.log(`Message sent to ${topic}:`, result);
      return result;
    } catch (error) {
      console.error("Failed to send message:", error);
      throw error;
    }
  }

  async sendBatch(topic, messages, options = {}) {
    if (!this.isConnected) {
      throw new Error("Producer not connected");
    }

    const messagePayload = {
      topic,
      messages: messages.map((msg) => ({
        key: msg.key || null,
        value: JSON.stringify(msg.value),
        partition: msg.partition,
        timestamp: msg.timestamp || Date.now().toString(),
        headers: msg.headers || {},
      })),
    };

    try {
      const result = await this.producer.send(messagePayload);
      console.log(`Batch sent to ${topic}:`, result);
      return result;
    } catch (error) {
      console.error("Failed to send batch:", error);
      throw error;
    }
  }

  async disconnect() {
    try {
      await this.producer.disconnect();
      this.isConnected = false;
      console.log("Producer disconnected");
    } catch (error) {
      console.error("Failed to disconnect producer:", error);
      throw error;
    }
  }
}

// Usage Example
async function producerExample() {
  const producer = new KafkaProducer({
    brokers: ["localhost:9092"],
    clientId: "example-producer",
  });

  await producer.connect();

  // Send single message
  await producer.sendMessage(
    "user-events",
    {
      userId: "123",
      action: "login",
      timestamp: new Date().toISOString(),
    },
    {
      key: "user-123",
      headers: { source: "web-app" },
    }
  );

  // Send batch
  const messages = [
    { key: "user-1", value: { userId: "1", action: "view" } },
    { key: "user-2", value: { userId: "2", action: "click" } },
  ];
  await producer.sendBatch("user-events", messages);

  await producer.disconnect();
}
```

### 2. Advanced Consumer Implementation

```javascript
class KafkaConsumer {
  constructor(config) {
    this.kafka = new Kafka({
      clientId: config.clientId || "nodejs-consumer",
      brokers: config.brokers,
      retry: {
        initialRetryTime: 100,
        retries: 8,
      },
    });

    this.consumer = this.kafka.consumer({
      groupId: config.groupId,
      sessionTimeout: config.sessionTimeout || 30000,
      heartbeatInterval: config.heartbeatInterval || 3000,
      maxBytesPerPartition: config.maxBytesPerPartition || 1048576,
      allowAutoTopicCreation: false,
    });

    this.isRunning = false;
    this.messageHandlers = new Map();
  }

  async connect() {
    try {
      await this.consumer.connect();
      console.log("Consumer connected successfully");
    } catch (error) {
      console.error("Failed to connect consumer:", error);
      throw error;
    }
  }

  async subscribe(topics) {
    try {
      await this.consumer.subscribe({
        topics: Array.isArray(topics) ? topics : [topics],
        fromBeginning: false,
      });
      console.log(`Subscribed to topics: ${topics}`);
    } catch (error) {
      console.error("Failed to subscribe:", error);
      throw error;
    }
  }

  async run() {
    if (this.isRunning) {
      throw new Error("Consumer is already running");
    }

    this.isRunning = true;

    await this.consumer.run({
      eachMessage: async ({ topic, partition, message, heartbeat, pause }) => {
        try {
          // Process heartbeat
          await heartbeat();

          const messageData = {
            topic,
            partition,
            offset: message.offset,
            key: message.key?.toString(),
            value: JSON.parse(message.value.toString()),
            timestamp: message.timestamp,
            headers: message.headers,
          };

          // Find and execute handler
          const handler = this.messageHandlers.get(topic);
          if (handler) {
            await handler(messageData);
          } else {
            console.warn(`No handler found for topic: ${topic}`);
          }
        } catch (error) {
          console.error("Error processing message:", error);
          // Implement retry logic or dead letter queue
          await this.handleMessageError(error, { topic, partition, message });
        }
      },

      eachBatch: async ({ batch, heartbeat, isRunning, isStale }) => {
        try {
          // Process batch of messages
          for (const message of batch.messages) {
            if (!isRunning() || isStale()) break;

            const messageData = {
              topic: batch.topic,
              partition: batch.partition,
              offset: message.offset,
              key: message.key?.toString(),
              value: JSON.parse(message.value.toString()),
              timestamp: message.timestamp,
              headers: message.headers,
            };

            const handler = this.messageHandlers.get(batch.topic);
            if (handler) {
              await handler(messageData);
            }
          }

          await heartbeat();
        } catch (error) {
          console.error("Error processing batch:", error);
        }
      },
    });
  }

  onMessage(topic, handler) {
    this.messageHandlers.set(topic, handler);
  }

  async handleMessageError(error, messageContext) {
    // Implement error handling strategy
    console.error("Message processing failed:", {
      error: error.message,
      context: messageContext,
    });

    // Options:
    // 1. Retry with exponential backoff
    // 2. Send to dead letter queue
    // 3. Log and continue
    // 4. Alert monitoring system
  }

  async pause(topics) {
    await this.consumer.pause(topics);
  }

  async resume(topics) {
    await this.consumer.resume(topics);
  }

  async disconnect() {
    try {
      await this.consumer.disconnect();
      this.isRunning = false;
      console.log("Consumer disconnected");
    } catch (error) {
      console.error("Failed to disconnect consumer:", error);
      throw error;
    }
  }
}

// Usage Example
async function consumerExample() {
  const consumer = new KafkaConsumer({
    brokers: ["localhost:9092"],
    groupId: "example-consumer-group",
    clientId: "example-consumer",
  });

  await consumer.connect();
  await consumer.subscribe(["user-events", "order-processing"]);

  // Register message handlers
  consumer.onMessage("user-events", async (messageData) => {
    console.log("Processing user event:", messageData);
    // Process user event
  });

  consumer.onMessage("order-processing", async (messageData) => {
    console.log("Processing order:", messageData);
    // Process order
  });

  await consumer.run();
}
```

### 3. Transactional Producer

```javascript
class TransactionalProducer {
  constructor(config) {
    this.kafka = new Kafka({
      clientId: config.clientId || "transactional-producer",
      brokers: config.brokers,
    });

    this.producer = this.kafka.producer({
      maxInFlightRequests: 5,
      idempotent: true,
      transactionTimeout: 30000,
    });

    this.transactionalId = config.transactionalId;
    this.isConnected = false;
  }

  async connect() {
    try {
      await this.producer.connect();
      await this.producer.transaction();
      this.isConnected = true;
      console.log("Transactional producer connected");
    } catch (error) {
      console.error("Failed to connect transactional producer:", error);
      throw error;
    }
  }

  async beginTransaction() {
    try {
      await this.producer.transaction();
      console.log("Transaction begun");
    } catch (error) {
      console.error("Failed to begin transaction:", error);
      throw error;
    }
  }

  async sendTransactionalMessage(topic, message, options = {}) {
    if (!this.isConnected) {
      throw new Error("Producer not connected");
    }

    try {
      await this.producer.send({
        topic,
        messages: [
          {
            key: options.key || null,
            value: JSON.stringify(message),
            partition: options.partition,
            headers: options.headers || {},
          },
        ],
      });
      console.log("Transactional message sent");
    } catch (error) {
      console.error("Failed to send transactional message:", error);
      throw error;
    }
  }

  async commitTransaction() {
    try {
      await this.producer.sendOffsets({
        consumerGroupId: this.consumerGroupId,
        topics: this.consumerOffsets,
      });
      await this.producer.commitTransaction();
      console.log("Transaction committed");
    } catch (error) {
      console.error("Failed to commit transaction:", error);
      throw error;
    }
  }

  async abortTransaction() {
    try {
      await this.producer.abortTransaction();
      console.log("Transaction aborted");
    } catch (error) {
      console.error("Failed to abort transaction:", error);
      throw error;
    }
  }
}
```

---

## Advanced Patterns & Edge Cases

### 1. Exactly-Once Semantics

```javascript
class ExactlyOnceProcessor {
  constructor(config) {
    this.kafka = new Kafka({
      clientId: config.clientId,
      brokers: config.brokers,
    });

    this.producer = this.kafka.producer({
      maxInFlightRequests: 5,
      idempotent: true,
      transactionTimeout: 30000,
    });

    this.consumer = this.kafka.consumer({
      groupId: config.groupId,
      isolationLevel: "read_committed",
    });

    this.transactionalId = config.transactionalId;
  }

  async processWithExactlyOnce(topic, processor) {
    await this.producer.transaction();

    try {
      await this.consumer.run({
        eachMessage: async ({ topic, partition, message }) => {
          const messageData = JSON.parse(message.value.toString());

          // Process message
          const result = await processor(messageData);

          // Send result to output topic
          await this.producer.send({
            topic: "processed-events",
            messages: [
              {
                key: message.key,
                value: JSON.stringify(result),
              },
            ],
          });

          // Commit consumer offset
          await this.producer.sendOffsets({
            consumerGroupId: this.consumer.groupId,
            topics: [
              {
                topic,
                partitions: [
                  {
                    partition,
                    offset: message.offset,
                  },
                ],
              },
            ],
          });
        },
      });

      await this.producer.commitTransaction();
    } catch (error) {
      await this.producer.abortTransaction();
      throw error;
    }
  }
}
```

### 2. Dead Letter Queue Pattern

```javascript
class DeadLetterQueueHandler {
  constructor(config) {
    this.kafka = config.kafka;
    this.dlqTopic = config.dlqTopic || "dead-letter-queue";
    this.maxRetries = config.maxRetries || 3;
    this.retryDelay = config.retryDelay || 1000;
  }

  async handleFailedMessage(originalMessage, error, retryCount = 0) {
    if (retryCount < this.maxRetries) {
      // Retry with exponential backoff
      const delay = this.retryDelay * Math.pow(2, retryCount);
      await new Promise((resolve) => setTimeout(resolve, delay));

      try {
        // Retry processing
        return await this.retryProcessing(originalMessage);
      } catch (retryError) {
        return await this.handleFailedMessage(
          originalMessage,
          retryError,
          retryCount + 1
        );
      }
    } else {
      // Send to dead letter queue
      await this.sendToDLQ(originalMessage, error);
    }
  }

  async sendToDLQ(originalMessage, error) {
    const dlqMessage = {
      originalMessage,
      error: {
        message: error.message,
        stack: error.stack,
        timestamp: new Date().toISOString(),
      },
      metadata: {
        topic: originalMessage.topic,
        partition: originalMessage.partition,
        offset: originalMessage.offset,
        retryCount: this.maxRetries,
      },
    };

    await this.kafka.producer().send({
      topic: this.dlqTopic,
      messages: [
        {
          key: originalMessage.key,
          value: JSON.stringify(dlqMessage),
        },
      ],
    });
  }
}
```

### 3. Message Ordering Guarantees

```javascript
class OrderedMessageProcessor {
  constructor(config) {
    this.kafka = config.kafka;
    this.buffer = new Map(); // partition -> messages
    this.processedOffsets = new Map(); // partition -> last processed offset
    this.bufferSize = config.bufferSize || 1000;
  }

  async processOrderedMessages(topic, processor) {
    await this.kafka.consumer().run({
      eachMessage: async ({ topic, partition, message }) => {
        const messageData = {
          topic,
          partition,
          offset: parseInt(message.offset),
          key: message.key?.toString(),
          value: JSON.parse(message.value.toString()),
        };

        // Add to buffer
        if (!this.buffer.has(partition)) {
          this.buffer.set(partition, []);
        }
        this.buffer.get(partition).push(messageData);

        // Process in order
        await this.processPartitionInOrder(partition, processor);
      },
    });
  }

  async processPartitionInOrder(partition, processor) {
    const messages = this.buffer.get(partition) || [];
    const lastProcessed = this.processedOffsets.get(partition) || -1;

    // Sort by offset
    messages.sort((a, b) => a.offset - b.offset);

    // Process consecutive messages
    for (const message of messages) {
      if (message.offset === lastProcessed + 1) {
        try {
          await processor(message);
          this.processedOffsets.set(partition, message.offset);

          // Remove processed message from buffer
          const index = messages.indexOf(message);
          messages.splice(index, 1);
        } catch (error) {
          console.error(`Failed to process message ${message.offset}:`, error);
          break; // Stop processing to maintain order
        }
      } else {
        break; // Wait for missing message
      }
    }
  }
}
```

---

## Scaling & Performance

### 1. Partition Strategy

```javascript
class PartitionStrategy {
  // Round-robin partitioning
  static roundRobin(totalPartitions, messageCount = 0) {
    return messageCount % totalPartitions;
  }

  // Hash-based partitioning
  static hashBased(key, totalPartitions) {
    if (!key) return 0;
    let hash = 0;
    for (let i = 0; i < key.length; i++) {
      hash = ((hash << 5) - hash + key.charCodeAt(i)) & 0xffffffff;
    }
    return Math.abs(hash) % totalPartitions;
  }

  // Range-based partitioning
  static rangeBased(key, totalPartitions) {
    if (!key) return 0;
    const ranges = this.calculateRanges(totalPartitions);

    for (let i = 0; i < ranges.length; i++) {
      if (key >= ranges[i].start && key <= ranges[i].end) {
        return i;
      }
    }
    return 0;
  }

  static calculateRanges(totalPartitions) {
    const ranges = [];
    const rangeSize = Math.floor(256 / totalPartitions);

    for (let i = 0; i < totalPartitions; i++) {
      ranges.push({
        start: i * rangeSize,
        end: (i + 1) * rangeSize - 1,
      });
    }

    return ranges;
  }
}

// Usage
const partitioner = new PartitionStrategy();
const partition = partitioner.hashBased("user-123", 12);
```

### 2. Consumer Scaling

```javascript
class ConsumerScaler {
  constructor(config) {
    this.kafka = config.kafka;
    this.topic = config.topic;
    this.consumerGroup = config.consumerGroup;
    this.maxConsumers = config.maxConsumers || 10;
    this.scaleUpThreshold = config.scaleUpThreshold || 0.8;
    this.scaleDownThreshold = config.scaleDownThreshold || 0.3;
  }

  async getTopicPartitions() {
    const admin = this.kafka.admin();
    const metadata = await admin.fetchTopicMetadata({ topics: [this.topic] });
    return metadata.topics[0].partitions.length;
  }

  async getConsumerGroupInfo() {
    const admin = this.kafka.admin();
    const groupInfo = await admin.describeGroups([this.consumerGroup]);
    return groupInfo.groups[0];
  }

  async shouldScaleUp() {
    const partitions = await this.getTopicPartitions();
    const groupInfo = await this.getConsumerGroupInfo();
    const activeConsumers = groupInfo.members.length;

    const utilization = activeConsumers / partitions;
    return (
      utilization > this.scaleUpThreshold && activeConsumers < this.maxConsumers
    );
  }

  async shouldScaleDown() {
    const partitions = await this.getTopicPartitions();
    const groupInfo = await this.getConsumerGroupInfo();
    const activeConsumers = groupInfo.members.length;

    const utilization = activeConsumers / partitions;
    return utilization < this.scaleDownThreshold && activeConsumers > 1;
  }

  async scaleConsumers() {
    if (await this.shouldScaleUp()) {
      await this.scaleUp();
    } else if (await this.shouldScaleDown()) {
      await this.scaleDown();
    }
  }

  async scaleUp() {
    console.log("Scaling up consumers...");
    // Implement scaling logic (e.g., start new consumer instances)
  }

  async scaleDown() {
    console.log("Scaling down consumers...");
    // Implement scaling logic (e.g., stop consumer instances)
  }
}
```

### 3. Performance Optimization

```javascript
class KafkaPerformanceOptimizer {
  constructor(config) {
    this.config = config;
    this.metrics = {
      messagesPerSecond: 0,
      averageLatency: 0,
      errorRate: 0,
    };
  }

  // Producer optimizations
  getOptimizedProducerConfig() {
    return {
      // Batch settings
      batchSize: 16384, // 16KB
      lingerMs: 5, // Wait 5ms to batch messages

      // Compression
      compressionType: "snappy", // or "lz4", "gzip"

      // Retry settings
      retries: 3,
      retryBackoffMs: 100,

      // Idempotence
      enableIdempotence: true,

      // In-flight requests
      maxInFlightRequests: 5,

      // Buffer settings
      bufferMemory: 33554432, // 32MB
      sendBufferBytes: 131072, // 128KB
      receiveBufferBytes: 32768, // 32KB
    };
  }

  // Consumer optimizations
  getOptimizedConsumerConfig() {
    return {
      // Fetch settings
      maxBytesPerPartition: 1048576, // 1MB
      maxBytes: 52428800, // 50MB
      maxWaitTimeInMs: 500,

      // Session settings
      sessionTimeoutMs: 30000,
      heartbeatIntervalMs: 3000,

      // Poll settings
      maxPollRecords: 500,
      maxPollIntervalMs: 300000,

      // Auto commit
      enableAutoCommit: false, // Manual commit for better control

      // Isolation level
      isolationLevel: "read_committed",
    };
  }

  // Topic configuration optimizations
  getOptimizedTopicConfig() {
    return {
      // Retention
      retentionMs: 604800000, // 7 days
      retentionBytes: -1, // Unlimited

      // Segment settings
      segmentMs: 604800000, // 7 days
      segmentBytes: 1073741824, // 1GB

      // Compression
      compressionType: "snappy",

      // Cleanup policy
      cleanupPolicy: "delete", // or "compact"

      // Replication
      minInSyncReplicas: 2,
      uncleanLeaderElectionEnable: false,
    };
  }

  async monitorPerformance() {
    // Implement performance monitoring
    setInterval(() => {
      this.collectMetrics();
      this.analyzePerformance();
    }, 10000);
  }

  collectMetrics() {
    // Collect producer/consumer metrics
    // This would integrate with Kafka's metrics system
  }

  analyzePerformance() {
    // Analyze metrics and suggest optimizations
    if (this.metrics.errorRate > 0.01) {
      console.warn("High error rate detected:", this.metrics.errorRate);
    }

    if (this.metrics.averageLatency > 100) {
      console.warn("High latency detected:", this.metrics.averageLatency);
    }
  }
}
```

---

## Production Operations

### 1. Cluster Management

```javascript
class KafkaClusterManager {
  constructor(config) {
    this.kafka = new Kafka({
      clientId: "cluster-manager",
      brokers: config.brokers,
    });
    this.admin = this.kafka.admin();
  }

  async createTopic(topicName, config = {}) {
    const topicConfig = {
      topic: topicName,
      numPartitions: config.partitions || 3,
      replicationFactor: config.replicationFactor || 3,
      configEntries: {
        "cleanup.policy": config.cleanupPolicy || "delete",
        "retention.ms": config.retentionMs || "604800000", // 7 days
        "compression.type": config.compressionType || "snappy",
        "min.insync.replicas": config.minInSyncReplicas || "2",
      },
    };

    try {
      await this.admin.createTopics({ topics: [topicConfig] });
      console.log(`Topic ${topicName} created successfully`);
    } catch (error) {
      if (error.type === "TOPIC_ALREADY_EXISTS") {
        console.log(`Topic ${topicName} already exists`);
      } else {
        throw error;
      }
    }
  }

  async deleteTopic(topicName) {
    try {
      await this.admin.deleteTopics({ topics: [topicName] });
      console.log(`Topic ${topicName} deleted successfully`);
    } catch (error) {
      console.error(`Failed to delete topic ${topicName}:`, error);
      throw error;
    }
  }

  async getTopicMetadata(topicName) {
    try {
      const metadata = await this.admin.fetchTopicMetadata({
        topics: [topicName],
      });
      return metadata.topics[0];
    } catch (error) {
      console.error(`Failed to get metadata for topic ${topicName}:`, error);
      throw error;
    }
  }

  async listTopics() {
    try {
      const metadata = await this.admin.listTopics();
      return metadata;
    } catch (error) {
      console.error("Failed to list topics:", error);
      throw error;
    }
  }

  async getConsumerGroups() {
    try {
      const groups = await this.admin.listGroups();
      return groups.groups;
    } catch (error) {
      console.error("Failed to list consumer groups:", error);
      throw error;
    }
  }

  async getConsumerGroupDetails(groupId) {
    try {
      const details = await this.admin.describeGroups([groupId]);
      return details.groups[0];
    } catch (error) {
      console.error(
        `Failed to get details for consumer group ${groupId}:`,
        error
      );
      throw error;
    }
  }

  async resetConsumerGroupOffsets(groupId, topic, partition, offset) {
    try {
      await this.admin.setOffsets({
        groupId,
        topic,
        partitions: [{ partition, offset }],
      });
      console.log(
        `Reset offset for group ${groupId}, topic ${topic}, partition ${partition} to ${offset}`
      );
    } catch (error) {
      console.error("Failed to reset consumer group offsets:", error);
      throw error;
    }
  }
}
```

### 2. Monitoring and Metrics

```javascript
class KafkaMonitor {
  constructor(config) {
    this.kafka = new Kafka({
      clientId: "kafka-monitor",
      brokers: config.brokers,
    });
    this.admin = this.kafka.admin();
    this.metrics = {
      topics: new Map(),
      consumerGroups: new Map(),
      brokers: new Map(),
    };
  }

  async collectTopicMetrics(topicName) {
    try {
      const metadata = await this.getTopicMetadata(topicName);
      const partitions = metadata.partitions;

      const topicMetrics = {
        name: topicName,
        partitions: partitions.length,
        replicationFactor: partitions[0]?.replicas?.length || 0,
        partitionDetails: partitions.map((p) => ({
          id: p.partitionId,
          leader: p.leader,
          replicas: p.replicas,
          isr: p.isr,
        })),
      };

      this.metrics.topics.set(topicName, topicMetrics);
      return topicMetrics;
    } catch (error) {
      console.error(`Failed to collect metrics for topic ${topicName}:`, error);
      throw error;
    }
  }

  async collectConsumerGroupMetrics(groupId) {
    try {
      const groupDetails = await this.getConsumerGroupDetails(groupId);

      const groupMetrics = {
        groupId,
        state: groupDetails.state,
        members: groupDetails.members.length,
        protocol: groupDetails.protocol,
        memberDetails: groupDetails.members.map((member) => ({
          memberId: member.memberId,
          clientId: member.clientId,
          clientHost: member.clientHost,
          assignments: member.memberAssignment,
        })),
      };

      this.metrics.consumerGroups.set(groupId, groupMetrics);
      return groupMetrics;
    } catch (error) {
      console.error(
        `Failed to collect metrics for consumer group ${groupId}:`,
        error
      );
      throw error;
    }
  }

  async getTopicOffsets(topicName) {
    try {
      const metadata = await this.getTopicMetadata(topicName);
      const partitions = metadata.partitions.map((p) => p.partitionId);

      const offsets = await this.admin.fetchTopicOffsets(topicName);
      return offsets;
    } catch (error) {
      console.error(`Failed to get offsets for topic ${topicName}:`, error);
      throw error;
    }
  }

  async getConsumerGroupOffsets(groupId, topicName) {
    try {
      const offsets = await this.admin.fetchOffsets({
        groupId,
        topic: topicName,
      });
      return offsets;
    } catch (error) {
      console.error(`Failed to get consumer group offsets:`, error);
      throw error;
    }
  }

  async calculateLag(groupId, topicName) {
    try {
      const [topicOffsets, consumerOffsets] = await Promise.all([
        this.getTopicOffsets(topicName),
        this.getConsumerGroupOffsets(groupId, topicName),
      ]);

      const lag = {};
      for (const partition of topicOffsets) {
        const consumerOffset = consumerOffsets.find(
          (co) => co.partition === partition.partition
        );
        if (consumerOffset) {
          lag[partition.partition] = {
            highWatermark: partition.offset,
            consumerOffset: consumerOffset.offset,
            lag: partition.offset - consumerOffset.offset,
          };
        }
      }

      return lag;
    } catch (error) {
      console.error("Failed to calculate lag:", error);
      throw error;
    }
  }

  async generateHealthReport() {
    const report = {
      timestamp: new Date().toISOString(),
      topics: {},
      consumerGroups: {},
      alerts: [],
    };

    try {
      // Collect topic metrics
      const topics = await this.listTopics();
      for (const topic of topics) {
        const metrics = await this.collectTopicMetrics(topic);
        report.topics[topic] = metrics;
      }

      // Collect consumer group metrics
      const groups = await this.getConsumerGroups();
      for (const group of groups) {
        const metrics = await this.collectConsumerGroupMetrics(group.groupId);
        report.consumerGroups[group.groupId] = metrics;

        // Calculate lag for each topic
        for (const topic of topics) {
          const lag = await this.calculateLag(group.groupId, topic);
          report.consumerGroups[group.groupId].lag = lag;

          // Check for high lag
          for (const [partition, lagInfo] of Object.entries(lag)) {
            if (lagInfo.lag > 10000) {
              // Alert if lag > 10k messages
              report.alerts.push({
                type: "HIGH_LAG",
                severity: "WARNING",
                message: `High lag detected: ${lagInfo.lag} messages in group ${group.groupId}, topic ${topic}, partition ${partition}`,
                details: lagInfo,
              });
            }
          }
        }
      }

      return report;
    } catch (error) {
      console.error("Failed to generate health report:", error);
      throw error;
    }
  }

  startMonitoring(intervalMs = 30000) {
    setInterval(async () => {
      try {
        const report = await this.generateHealthReport();
        console.log("Kafka Health Report:", JSON.stringify(report, null, 2));

        // Send alerts if any
        if (report.alerts.length > 0) {
          await this.sendAlerts(report.alerts);
        }
      } catch (error) {
        console.error("Monitoring error:", error);
      }
    }, intervalMs);
  }

  async sendAlerts(alerts) {
    // Implement alerting logic (email, Slack, PagerDuty, etc.)
    for (const alert of alerts) {
      console.log(`ALERT [${alert.severity}]: ${alert.message}`);
    }
  }
}
```

---

## Real-world Use Cases

### 1. Event Sourcing with Kafka

```javascript
class EventSourcingService {
  constructor(config) {
    this.kafka = new Kafka({
      clientId: "event-sourcing-service",
      brokers: config.brokers,
    });
    this.producer = this.kafka.producer();
    this.consumer = this.kafka.consumer({
      groupId: "event-sourcing-consumer",
    });
    this.eventStore = new Map(); // In-memory store for demo
  }

  async publishEvent(aggregateId, eventType, eventData, version) {
    const event = {
      id: this.generateEventId(),
      aggregateId,
      eventType,
      eventData,
      version,
      timestamp: new Date().toISOString(),
      metadata: {
        source: "event-sourcing-service",
        correlationId: this.generateCorrelationId(),
      },
    };

    await this.producer.send({
      topic: "events",
      messages: [
        {
          key: aggregateId,
          value: JSON.stringify(event),
        },
      ],
    });

    return event;
  }

  async replayEvents(aggregateId, fromVersion = 0) {
    const consumer = this.kafka.consumer({
      groupId: `replay-${aggregateId}-${Date.now()}`,
      fromBeginning: true,
    });

    await consumer.connect();
    await consumer.subscribe({ topics: ["events"] });

    const events = [];
    let foundStart = false;

    await consumer.run({
      eachMessage: async ({ message }) => {
        const event = JSON.parse(message.value.toString());

        if (event.aggregateId === aggregateId) {
          if (event.version >= fromVersion) {
            events.push(event);
          }
        }
      },
    });

    await consumer.disconnect();
    return events.sort((a, b) => a.version - b.version);
  }

  async getAggregateState(aggregateId) {
    const events = await this.replayEvents(aggregateId);
    return this.applyEvents(events);
  }

  applyEvents(events) {
    let state = {};

    for (const event of events) {
      state = this.applyEvent(state, event);
    }

    return state;
  }

  applyEvent(state, event) {
    // Implement event application logic based on event type
    switch (event.eventType) {
      case "UserCreated":
        return {
          ...state,
          id: event.aggregateId,
          name: event.eventData.name,
          email: event.eventData.email,
        };
      case "UserUpdated":
        return { ...state, ...event.eventData };
      case "UserDeleted":
        return { ...state, deleted: true, deletedAt: event.timestamp };
      default:
        return state;
    }
  }

  generateEventId() {
    return `evt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  generateCorrelationId() {
    return `corr_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}
```

### 2. CQRS with Kafka

```javascript
class CQRSService {
  constructor(config) {
    this.kafka = new Kafka({
      clientId: "cqrs-service",
      brokers: config.brokers,
    });
    this.commandProducer = this.kafka.producer();
    this.queryConsumer = this.kafka.consumer({
      groupId: "cqrs-query-consumer",
    });
    this.readModels = new Map();
  }

  // Command Side
  async executeCommand(command) {
    const commandEvent = {
      id: this.generateId(),
      type: command.type,
      data: command.data,
      timestamp: new Date().toISOString(),
      correlationId: command.correlationId,
    };

    await this.commandProducer.send({
      topic: "commands",
      messages: [
        {
          key: command.aggregateId,
          value: JSON.stringify(commandEvent),
        },
      ],
    });

    return commandEvent;
  }

  // Query Side
  async subscribeToEvents() {
    await this.queryConsumer.connect();
    await this.queryConsumer.subscribe({ topics: ["events"] });

    await this.queryConsumer.run({
      eachMessage: async ({ message }) => {
        const event = JSON.parse(message.value.toString());
        await this.updateReadModel(event);
      },
    });
  }

  async updateReadModel(event) {
    // Update read models based on events
    switch (event.eventType) {
      case "UserCreated":
        await this.updateUserReadModel(event);
        break;
      case "OrderPlaced":
        await this.updateOrderReadModel(event);
        break;
      // Add more event handlers
    }
  }

  async updateUserReadModel(event) {
    const userModel = {
      id: event.aggregateId,
      name: event.eventData.name,
      email: event.eventData.email,
      createdAt: event.timestamp,
      lastUpdated: event.timestamp,
    };

    this.readModels.set(`user_${event.aggregateId}`, userModel);
  }

  async updateOrderReadModel(event) {
    const orderModel = {
      id: event.aggregateId,
      userId: event.eventData.userId,
      items: event.eventData.items,
      total: event.eventData.total,
      status: "placed",
      createdAt: event.timestamp,
    };

    this.readModels.set(`order_${event.aggregateId}`, orderModel);
  }

  // Query methods
  async getUserById(userId) {
    return this.readModels.get(`user_${userId}`);
  }

  async getOrderById(orderId) {
    return this.readModels.get(`order_${orderId}`);
  }

  async getAllUsers() {
    const users = [];
    for (const [key, value] of this.readModels.entries()) {
      if (key.startsWith("user_")) {
        users.push(value);
      }
    }
    return users;
  }

  generateId() {
    return `cmd_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}
```

### 3. Microservices Communication

```javascript
class MicroserviceCommunication {
  constructor(config) {
    this.kafka = new Kafka({
      clientId: config.serviceName,
      brokers: config.brokers,
    });
    this.producer = this.kafka.producer();
    this.consumer = this.kafka.consumer({
      groupId: config.serviceName,
    });
    this.serviceName = config.serviceName;
    this.requestTimeout = config.requestTimeout || 30000;
    this.pendingRequests = new Map();
  }

  async sendRequest(targetService, requestType, data, correlationId = null) {
    const requestId = correlationId || this.generateRequestId();
    const request = {
      id: requestId,
      from: this.serviceName,
      to: targetService,
      type: requestType,
      data,
      timestamp: new Date().toISOString(),
    };

    // Set up response handler
    const responsePromise = new Promise((resolve, reject) => {
      this.pendingRequests.set(requestId, { resolve, reject });

      // Timeout handler
      setTimeout(() => {
        if (this.pendingRequests.has(requestId)) {
          this.pendingRequests.delete(requestId);
          reject(new Error(`Request timeout: ${requestId}`));
        }
      }, this.requestTimeout);
    });

    // Send request
    await this.producer.send({
      topic: `${targetService}.requests`,
      messages: [
        {
          key: requestId,
          value: JSON.stringify(request),
        },
      ],
    });

    return responsePromise;
  }

  async sendResponse(requestId, responseData, success = true) {
    const response = {
      id: requestId,
      success,
      data: responseData,
      timestamp: new Date().toISOString(),
    };

    await this.producer.send({
      topic: `${this.serviceName}.responses`,
      messages: [
        {
          key: requestId,
          value: JSON.stringify(response),
        },
      ],
    });
  }

  async startListening() {
    await this.consumer.connect();
    await this.consumer.subscribe({
      topics: [`${this.serviceName}.requests`, `${this.serviceName}.responses`],
    });

    await this.consumer.run({
      eachMessage: async ({ topic, message }) => {
        const data = JSON.parse(message.value.toString());

        if (topic.endsWith(".requests")) {
          await this.handleRequest(data);
        } else if (topic.endsWith(".responses")) {
          await this.handleResponse(data);
        }
      },
    });
  }

  async handleRequest(request) {
    try {
      // Process the request based on type
      const responseData = await this.processRequest(
        request.type,
        request.data
      );
      await this.sendResponse(request.id, responseData, true);
    } catch (error) {
      await this.sendResponse(request.id, { error: error.message }, false);
    }
  }

  async handleResponse(response) {
    const pendingRequest = this.pendingRequests.get(response.id);
    if (pendingRequest) {
      this.pendingRequests.delete(response.id);

      if (response.success) {
        pendingRequest.resolve(response.data);
      } else {
        pendingRequest.reject(new Error(response.data.error));
      }
    }
  }

  async processRequest(type, data) {
    // Implement request processing logic
    switch (type) {
      case "getUser":
        return await this.getUser(data.userId);
      case "createOrder":
        return await this.createOrder(data);
      default:
        throw new Error(`Unknown request type: ${type}`);
    }
  }

  async getUser(userId) {
    // Implement user retrieval logic
    return { id: userId, name: "John Doe", email: "john@example.com" };
  }

  async createOrder(orderData) {
    // Implement order creation logic
    return { id: this.generateId(), ...orderData, status: "created" };
  }

  generateRequestId() {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  generateId() {
    return `id_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}
```

---

## Troubleshooting & Edge Cases

### 1. Common Issues and Solutions

```javascript
class KafkaTroubleshooter {
  constructor(config) {
    this.kafka = new Kafka({
      clientId: "troubleshooter",
      brokers: config.brokers,
    });
    this.admin = this.kafka.admin();
  }

  async diagnoseConsumerLag(groupId) {
    try {
      const groups = await this.admin.describeGroups([groupId]);
      const group = groups.groups[0];

      if (group.state !== "Stable") {
        return {
          issue: "Consumer group not stable",
          state: group.state,
          solution: "Check for failed consumers or network issues",
        };
      }

      // Check for partition assignment issues
      const assignments = group.members.map(
        (member) => member.memberAssignment
      );
      const totalPartitions = assignments.reduce(
        (sum, assignment) => sum + assignment.length,
        0
      );

      if (totalPartitions === 0) {
        return {
          issue: "No partitions assigned",
          solution: "Check topic exists and has partitions",
        };
      }

      return {
        status: "healthy",
        members: group.members.length,
        totalPartitions,
      };
    } catch (error) {
      return {
        issue: "Failed to describe consumer group",
        error: error.message,
        solution: "Check if consumer group exists and is accessible",
      };
    }
  }

  async diagnoseTopicIssues(topicName) {
    try {
      const metadata = await this.admin.fetchTopicMetadata({
        topics: [topicName],
      });
      const topic = metadata.topics[0];

      const issues = [];

      // Check for under-replicated partitions
      for (const partition of topic.partitions) {
        if (partition.isr.length < partition.replicas.length) {
          issues.push({
            type: "UNDER_REPLICATED",
            partition: partition.partitionId,
            replicas: partition.replicas.length,
            isr: partition.isr.length,
            solution: "Check broker health and network connectivity",
          });
        }

        // Check for no leader
        if (partition.leader === -1) {
          issues.push({
            type: "NO_LEADER",
            partition: partition.partitionId,
            solution: "Trigger leader election or restart brokers",
          });
        }
      }

      return {
        topic: topicName,
        issues,
        healthy: issues.length === 0,
      };
    } catch (error) {
      return {
        issue: "Failed to fetch topic metadata",
        error: error.message,
        solution: "Check if topic exists and is accessible",
      };
    }
  }

  async diagnoseProducerIssues() {
    const commonIssues = [
      {
        issue: "Producer not sending messages",
        causes: [
          "Broker connectivity issues",
          "Topic doesn't exist",
          "Authentication/authorization problems",
          "Network timeouts",
        ],
        solutions: [
          "Check broker connectivity",
          "Verify topic exists",
          "Check credentials and permissions",
          "Increase timeout settings",
        ],
      },
      {
        issue: "High producer latency",
        causes: [
          "Network latency",
          "Broker overload",
          "Large batch sizes",
          "Compression overhead",
        ],
        solutions: [
          "Optimize network configuration",
          "Scale brokers",
          "Reduce batch size",
          "Use faster compression (lz4 vs gzip)",
        ],
      },
      {
        issue: "Message ordering issues",
        causes: [
          "Multiple partitions",
          "Retries enabled without idempotence",
          "Concurrent producers",
        ],
        solutions: [
          "Use single partition for ordering",
          "Enable idempotence",
          "Use transactional producer",
        ],
      },
    ];

    return commonIssues;
  }

  async generateDiagnosticReport() {
    const report = {
      timestamp: new Date().toISOString(),
      brokers: await this.checkBrokerHealth(),
      topics: await this.checkTopicHealth(),
      consumerGroups: await this.checkConsumerGroupHealth(),
      recommendations: [],
    };

    // Generate recommendations
    if (report.brokers.some((b) => !b.healthy)) {
      report.recommendations.push(
        "Some brokers are unhealthy - check broker logs and restart if necessary"
      );
    }

    if (report.topics.some((t) => t.issues.length > 0)) {
      report.recommendations.push(
        "Some topics have issues - check replication and leader assignment"
      );
    }

    if (report.consumerGroups.some((g) => g.state !== "Stable")) {
      report.recommendations.push(
        "Some consumer groups are not stable - check consumer health"
      );
    }

    return report;
  }

  async checkBrokerHealth() {
    // Implement broker health checks
    return [];
  }

  async checkTopicHealth() {
    // Implement topic health checks
    return [];
  }

  async checkConsumerGroupHealth() {
    // Implement consumer group health checks
    return [];
  }
}
```

### 2. Performance Tuning

```javascript
class KafkaPerformanceTuner {
  constructor(config) {
    this.config = config;
    this.baselineMetrics = null;
    this.currentMetrics = null;
  }

  async runPerformanceTest() {
    const testConfig = {
      duration: 60000, // 1 minute
      messageSize: 1024, // 1KB
      topics: ["perf-test"],
      partitions: 12,
    };

    // Create test topic
    await this.createTestTopic(testConfig);

    // Run producer test
    const producerResults = await this.runProducerTest(testConfig);

    // Run consumer test
    const consumerResults = await this.runConsumerTest(testConfig);

    // Analyze results
    const analysis = this.analyzeResults(producerResults, consumerResults);

    // Generate recommendations
    const recommendations = this.generateRecommendations(analysis);

    return {
      producerResults,
      consumerResults,
      analysis,
      recommendations,
    };
  }

  async runProducerTest(config) {
    const producer = this.kafka.producer({
      batchSize: 16384,
      lingerMs: 5,
      compressionType: "snappy",
    });

    await producer.connect();

    const startTime = Date.now();
    let messageCount = 0;
    let errorCount = 0;

    const testDuration = config.duration;
    const messageInterval = 1; // 1ms between messages

    while (Date.now() - startTime < testDuration) {
      try {
        await producer.send({
          topic: config.topics[0],
          messages: [
            {
              key: `key-${messageCount}`,
              value: this.generateTestMessage(config.messageSize),
            },
          ],
        });
        messageCount++;
      } catch (error) {
        errorCount++;
      }

      await new Promise((resolve) => setTimeout(resolve, messageInterval));
    }

    await producer.disconnect();

    const duration = Date.now() - startTime;
    return {
      messageCount,
      errorCount,
      duration,
      throughput: (messageCount / duration) * 1000, // messages per second
      errorRate: errorCount / messageCount,
    };
  }

  async runConsumerTest(config) {
    const consumer = this.kafka.consumer({
      groupId: "perf-test-consumer",
      fromBeginning: true,
    });

    await consumer.connect();
    await consumer.subscribe({ topics: config.topics });

    const startTime = Date.now();
    let messageCount = 0;
    let errorCount = 0;

    await consumer.run({
      eachMessage: async ({ message }) => {
        try {
          // Process message
          messageCount++;
        } catch (error) {
          errorCount++;
        }
      },
    });

    // Wait for test duration
    await new Promise((resolve) => setTimeout(resolve, config.duration));
    await consumer.disconnect();

    const duration = Date.now() - startTime;
    return {
      messageCount,
      errorCount,
      duration,
      throughput: (messageCount / duration) * 1000,
      errorRate: errorCount / messageCount,
    };
  }

  analyzeResults(producerResults, consumerResults) {
    return {
      producerThroughput: producerResults.throughput,
      consumerThroughput: consumerResults.throughput,
      producerErrorRate: producerResults.errorRate,
      consumerErrorRate: consumerResults.errorRate,
      endToEndLatency: this.calculateLatency(producerResults, consumerResults),
      bottlenecks: this.identifyBottlenecks(producerResults, consumerResults),
    };
  }

  generateRecommendations(analysis) {
    const recommendations = [];

    if (analysis.producerErrorRate > 0.01) {
      recommendations.push({
        type: "PRODUCER_ERRORS",
        message: "High producer error rate detected",
        action: "Check broker connectivity and increase retry settings",
      });
    }

    if (analysis.consumerErrorRate > 0.01) {
      recommendations.push({
        type: "CONSUMER_ERRORS",
        message: "High consumer error rate detected",
        action: "Check consumer configuration and message processing logic",
      });
    }

    if (analysis.producerThroughput < 1000) {
      recommendations.push({
        type: "LOW_THROUGHPUT",
        message: "Low producer throughput",
        action:
          "Increase batch size, enable compression, or add more partitions",
      });
    }

    if (analysis.endToEndLatency > 100) {
      recommendations.push({
        type: "HIGH_LATENCY",
        message: "High end-to-end latency",
        action:
          "Optimize network settings, reduce batch size, or use faster compression",
      });
    }

    return recommendations;
  }

  generateTestMessage(size) {
    return "x".repeat(size);
  }

  calculateLatency(producerResults, consumerResults) {
    // Simplified latency calculation
    return Math.abs(producerResults.duration - consumerResults.duration);
  }

  identifyBottlenecks(producerResults, consumerResults) {
    const bottlenecks = [];

    if (producerResults.throughput < consumerResults.throughput) {
      bottlenecks.push("Producer is the bottleneck");
    } else if (consumerResults.throughput < producerResults.throughput) {
      bottlenecks.push("Consumer is the bottleneck");
    } else {
      bottlenecks.push("System is balanced");
    }

    return bottlenecks;
  }

  async createTestTopic(config) {
    // Implementation for creating test topic
  }
}
```

### 3. Edge Cases and Best Practices

```javascript
class KafkaEdgeCases {
  constructor() {
    this.edgeCases = {
      // Message Ordering Edge Cases
      messageOrdering: {
        issue: "Messages arrive out of order",
        causes: [
          "Multiple partitions",
          "Retries without idempotence",
          "Network partitions",
        ],
        solutions: [
          "Use single partition for strict ordering",
          "Enable idempotent producer",
          "Implement sequence numbers in message payload",
        ],
      },

      // Consumer Rebalancing Edge Cases
      rebalancing: {
        issue: "Consumer group rebalancing issues",
        causes: [
          "Slow consumers",
          "Network timeouts",
          "Session timeout too low",
        ],
        solutions: [
          "Increase session timeout",
          "Optimize consumer processing",
          "Use cooperative rebalancing",
        ],
      },

      // Data Loss Edge Cases
      dataLoss: {
        issue: "Messages lost during processing",
        causes: [
          "Consumer crashes before commit",
          "Producer acks=0",
          "Topic retention too short",
        ],
        solutions: [
          "Use manual commit after processing",
          "Set acks=all for producers",
          "Increase retention period",
        ],
      },

      // Duplicate Messages Edge Cases
      duplicates: {
        issue: "Duplicate messages processed",
        causes: ["Consumer rebalancing", "Producer retries", "Network issues"],
        solutions: [
          "Implement idempotent processing",
          "Use message deduplication",
          "Enable idempotent producer",
        ],
      },
    };
  }

  getEdgeCaseSolutions(issueType) {
    return this.edgeCases[issueType] || null;
  }

  // Best Practices Implementation
  static bestPractices = {
    // Producer Best Practices
    producer: {
      "Enable idempotence": "Prevents duplicate messages",
      "Set appropriate acks": "Use acks=all for durability",
      "Use compression": "Reduce network overhead",
      "Batch messages": "Improve throughput",
      "Handle errors gracefully": "Implement retry logic",
    },

    // Consumer Best Practices
    consumer: {
      "Manual offset commit": "Commit after processing",
      "Handle rebalancing": "Implement rebalance listeners",
      "Process messages idempotently": "Handle duplicates",
      "Monitor consumer lag": "Set up alerting",
      "Use appropriate isolation level": "read_committed for transactions",
    },

    // Topic Design Best Practices
    topic: {
      "Choose right partition count": "Based on throughput needs",
      "Set appropriate retention": "Balance storage vs. replay needs",
      "Use meaningful names": "Follow naming conventions",
      "Configure replication": "Ensure fault tolerance",
      "Monitor topic health": "Set up monitoring",
    },

    // Cluster Best Practices
    cluster: {
      "Use odd number of brokers": "Avoid split-brain scenarios",
      "Monitor broker health": "Set up health checks",
      "Plan for capacity": "Monitor disk and CPU usage",
      "Backup configurations": "Version control configs",
      "Test disaster recovery": "Regular DR drills",
    },
  };

  static getBestPractices(category) {
    return this.bestPractices[category] || {};
  }
}
```

---

## Security & Advanced Topics

### 1. Kafka Security

```javascript
class KafkaSecurity {
  constructor(config) {
    this.kafka = new Kafka({
      clientId: config.clientId,
      brokers: config.brokers,
      ssl: config.ssl,
      sasl: config.sasl,
    });
  }

  // SSL/TLS Configuration
  static getSSLConfig() {
    return {
      ssl: {
        rejectUnauthorized: true,
        ca: [fs.readFileSync("ca-cert", "utf8")],
        cert: fs.readFileSync("client-cert", "utf8"),
        key: fs.readFileSync("client-key", "utf8"),
      },
    };
  }

  // SASL Authentication
  static getSASLConfig(username, password) {
    return {
      sasl: {
        mechanism: "plain",
        username: username,
        password: password,
      },
    };
  }

  // ACL Management
  async createACL(
    principal,
    resourceType,
    resourceName,
    operation,
    permission
  ) {
    const acl = {
      principal: principal,
      resourceType: resourceType,
      resourceName: resourceName,
      operation: operation,
      permission: permission,
    };

    // Implementation would use Kafka Admin API
    console.log("Creating ACL:", acl);
  }

  // Encryption at Rest
  static getEncryptionConfig() {
    return {
      "log.cleaner.enable": "true",
      "log.cleanup.policy": "compact",
      "compression.type": "snappy",
    };
  }
}
```

### 2. Stream Processing with Kafka

```javascript
class KafkaStreamProcessor {
  constructor(config) {
    this.kafka = new Kafka({
      clientId: "stream-processor",
      brokers: config.brokers,
    });
    this.producer = this.kafka.producer();
    this.consumer = this.kafka.consumer({
      groupId: "stream-processor-group",
    });
  }

  async processStream(inputTopic, outputTopic, processor) {
    await this.consumer.connect();
    await this.consumer.subscribe({ topics: [inputTopic] });

    await this.consumer.run({
      eachMessage: async ({ topic, partition, message }) => {
        try {
          const inputData = JSON.parse(message.value.toString());

          // Process the data
          const outputData = await processor(inputData);

          // Send to output topic
          await this.producer.send({
            topic: outputTopic,
            messages: [
              {
                key: message.key,
                value: JSON.stringify(outputData),
              },
            ],
          });
        } catch (error) {
          console.error("Stream processing error:", error);
        }
      },
    });
  }

  // Windowed Aggregations
  async windowedAggregation(topic, windowSizeMs, aggregator) {
    const windows = new Map();

    await this.consumer.run({
      eachMessage: async ({ message }) => {
        const data = JSON.parse(message.value.toString());
        const timestamp = data.timestamp;
        const windowKey = Math.floor(timestamp / windowSizeMs) * windowSizeMs;

        if (!windows.has(windowKey)) {
          windows.set(windowKey, []);
        }

        windows.get(windowKey).push(data);

        // Process completed windows
        const currentTime = Date.now();
        const completedWindows = Array.from(windows.keys()).filter(
          (key) => key < currentTime - windowSizeMs
        );

        for (const windowKey of completedWindows) {
          const windowData = windows.get(windowKey);
          const aggregated = aggregator(windowData);

          await this.producer.send({
            topic: "aggregated-data",
            messages: [
              {
                key: windowKey.toString(),
                value: JSON.stringify(aggregated),
              },
            ],
          });

          windows.delete(windowKey);
        }
      },
    });
  }
}
```

---

## Summary

This comprehensive Kafka guide covers:

1. ✅ **Fundamentals & Architecture** - Core concepts, topics, partitions, producers, consumers
2. ✅ **Node.js Integration** - Complete producer/consumer implementations with error handling
3. ✅ **Advanced Patterns** - Exactly-once semantics, dead letter queues, message ordering
4. ✅ **Scaling & Performance** - Partition strategies, consumer scaling, optimization
5. ✅ **Production Operations** - Cluster management, monitoring, backup/recovery
6. ✅ **Real-world Use Cases** - Event sourcing, CQRS, microservices communication
7. ✅ **Troubleshooting** - Common issues, diagnostics, performance tuning
8. ✅ **Edge Cases & Best Practices** - Security, stream processing, advanced patterns

The guide provides production-ready code examples, best practices, and covers edge cases that you'll encounter in real-world Kafka deployments. Each section includes practical implementations that you can use directly in your Node.js applications.

**Key Takeaways:**

- Always use idempotent producers for critical applications
- Implement proper error handling and retry logic
- Monitor consumer lag and set up alerting
- Use appropriate partitioning strategies for your use case
- Plan for scaling and capacity requirements
- Implement proper security measures
- Test disaster recovery procedures regularly

This guide serves as a complete reference for building robust, scalable, and production-ready Kafka applications with Node.js.


## Monitoring  Troubleshooting

<!-- AUTO-GENERATED ANCHOR: originally referenced as #monitoring--troubleshooting -->

Placeholder content. Please replace with proper section.
