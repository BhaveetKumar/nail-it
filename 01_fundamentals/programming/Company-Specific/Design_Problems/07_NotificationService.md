---
# Auto-generated front matter
Title: 07 Notificationservice
LastUpdated: 2025-11-06T20:45:58.774138
Tags: []
Status: draft
---

# 07. Notification Service - Multi-Channel Communication System

## Title & Summary
Design and implement a comprehensive notification service using Node.js that supports multiple channels (email, SMS, push, webhook) with templating, scheduling, and delivery tracking.

## Problem Statement

Build a robust notification system that:

1. **Multi-Channel Support**: Email, SMS, push notifications, webhooks
2. **Template Management**: Dynamic message templating
3. **Scheduling**: Delayed and recurring notifications
4. **Delivery Tracking**: Real-time delivery status and analytics
5. **User Preferences**: Channel preferences and opt-out management
6. **Rate Limiting**: Prevent spam and abuse

## Requirements & Constraints

### Functional Requirements
- Send notifications via multiple channels
- Template-based message generation
- Scheduled and recurring notifications
- Delivery status tracking
- User preference management
- Notification history and analytics
- Bulk notification support

### Non-Functional Requirements
- **Latency**: < 500ms for notification queuing
- **Throughput**: 10,000+ notifications per minute
- **Availability**: 99.9% uptime
- **Scalability**: Handle 1M+ notifications per day
- **Reliability**: 99.95% delivery success rate
- **Security**: Secure message transmission

## API / Interfaces

### REST Endpoints

```javascript
// Notification Management
POST   /api/notifications/send
POST   /api/notifications/bulk
GET    /api/notifications/{notificationId}
GET    /api/notifications
PUT    /api/notifications/{notificationId}/cancel

// Templates
POST   /api/templates
GET    /api/templates
PUT    /api/templates/{templateId}
DELETE /api/templates/{templateId}

// Scheduling
POST   /api/notifications/schedule
GET    /api/notifications/scheduled
PUT    /api/notifications/{notificationId}/reschedule

// User Preferences
GET    /api/users/{userId}/preferences
PUT    /api/users/{userId}/preferences
POST   /api/users/{userId}/opt-out

// Analytics
GET    /api/notifications/{notificationId}/status
GET    /api/analytics/delivery
GET    /api/analytics/channels
```

### Request/Response Examples

```json
// Send Notification
POST /api/notifications/send
{
  "templateId": "welcome_email",
  "recipients": [
    {
      "userId": "user_123",
      "email": "user@example.com",
      "phone": "+1234567890"
    }
  ],
  "variables": {
    "name": "John Doe",
    "company": "Acme Corp"
  },
  "channels": ["email", "sms"],
  "priority": "high",
  "scheduleAt": "2024-01-16T09:00:00Z"
}

// Response
{
  "success": true,
  "data": {
    "notificationId": "notif_456",
    "status": "queued",
    "recipients": 1,
    "channels": ["email", "sms"],
    "scheduledAt": "2024-01-16T09:00:00Z",
    "createdAt": "2024-01-15T10:30:00Z"
  }
}

// Notification Status
{
  "success": true,
  "data": {
    "notificationId": "notif_456",
    "status": "delivered",
    "totalRecipients": 1,
    "deliveryStats": {
      "email": {
        "sent": 1,
        "delivered": 1,
        "failed": 0,
        "pending": 0
      },
      "sms": {
        "sent": 1,
        "delivered": 1,
        "failed": 0,
        "pending": 0
      }
    },
    "createdAt": "2024-01-15T10:30:00Z",
    "completedAt": "2024-01-15T10:30:05Z"
  }
}
```

## Data Model

### Core Entities

```javascript
// Notification Entity
class Notification {
  constructor(templateId, recipients, variables, channels) {
    this.id = this.generateID();
    this.templateId = templateId;
    this.recipients = recipients;
    this.variables = variables;
    this.channels = channels;
    this.priority = "normal"; // 'low', 'normal', 'high', 'urgent'
    this.status = "queued"; // 'queued', 'processing', 'sent', 'delivered', 'failed'
    this.scheduledAt = new Date();
    this.createdAt = new Date();
    this.updatedAt = new Date();
    this.completedAt = null;
    this.deliveryStats = {};
    this.error = null;
  }
}

// Template Entity
class Template {
  constructor(name, type, content) {
    this.id = this.generateID();
    this.name = name;
    this.type = type; // 'email', 'sms', 'push', 'webhook'
    this.subject = "";
    this.content = content;
    this.variables = [];
    this.isActive = true;
    this.createdAt = new Date();
    this.updatedAt = new Date();
    this.createdBy = null;
  }
}

// Delivery Entity
class Delivery {
  constructor(notificationId, recipientId, channel, message) {
    this.id = this.generateID();
    this.notificationId = notificationId;
    this.recipientId = recipientId;
    this.channel = channel;
    this.message = message;
    this.status = "pending"; // 'pending', 'sent', 'delivered', 'failed', 'bounced'
    this.providerId = null;
    this.providerResponse = null;
    this.sentAt = null;
    this.deliveredAt = null;
    this.failedAt = null;
    this.error = null;
    this.retryCount = 0;
    this.maxRetries = 3;
  }
}

// User Preferences Entity
class UserPreferences {
  constructor(userId) {
    this.userId = userId;
    this.channels = {
      email: { enabled: true, frequency: "immediate" },
      sms: { enabled: true, frequency: "immediate" },
      push: { enabled: true, frequency: "immediate" },
      webhook: { enabled: false, frequency: "immediate" }
    };
    this.categories = {
      marketing: { enabled: true },
      transactional: { enabled: true },
      security: { enabled: true },
      system: { enabled: true }
    };
    this.quietHours = {
      enabled: false,
      start: "22:00",
      end: "08:00",
      timezone: "UTC"
    };
    this.updatedAt = new Date();
  }
}

// Channel Provider Entity
class ChannelProvider {
  constructor(name, type, config) {
    this.name = name;
    this.type = type; // 'email', 'sms', 'push', 'webhook'
    this.config = config;
    this.isActive = true;
    this.rateLimit = {
      requests: 1000,
      window: 3600000 // 1 hour
    };
    this.lastUsed = null;
  }
}
```

## Approach Overview

### Simple Solution (MVP)
1. In-memory notification storage
2. Basic email and SMS sending
3. Simple template system
4. No scheduling or analytics

### Production-Ready Design
1. **Multi-Channel Architecture**: Support for all notification types
2. **Template Engine**: Dynamic message generation
3. **Queue System**: Reliable message delivery
4. **Provider Management**: Multiple service providers
5. **Analytics**: Comprehensive delivery tracking
6. **Rate Limiting**: Prevent abuse and spam

## Detailed Design

### Core Service Implementation

```javascript
const EventEmitter = require("events");
const { v4: uuidv4 } = require("uuid");

class NotificationService extends EventEmitter {
  constructor() {
    super();
    this.notifications = new Map();
    this.templates = new Map();
    this.deliveries = new Map();
    this.userPreferences = new Map();
    this.providers = new Map();
    this.deliveryQueue = [];
    this.isProcessing = false;
    
    // Initialize providers
    this.initializeProviders();
    
    // Start background tasks
    this.startDeliveryProcessor();
    this.startScheduler();
    this.startCleanupTask();
  }

  initializeProviders() {
    // Email providers
    this.providers.set("email_smtp", new ChannelProvider("SMTP", "email", {
      host: process.env.SMTP_HOST,
      port: process.env.SMTP_PORT,
      username: process.env.SMTP_USERNAME,
      password: process.env.SMTP_PASSWORD
    }));
    
    this.providers.set("email_sendgrid", new ChannelProvider("SendGrid", "email", {
      apiKey: process.env.SENDGRID_API_KEY
    }));
    
    // SMS providers
    this.providers.set("sms_twilio", new ChannelProvider("Twilio", "sms", {
      accountSid: process.env.TWILIO_ACCOUNT_SID,
      authToken: process.env.TWILIO_AUTH_TOKEN,
      fromNumber: process.env.TWILIO_FROM_NUMBER
    }));
    
    // Push notification providers
    this.providers.set("push_fcm", new ChannelProvider("Firebase", "push", {
      serverKey: process.env.FCM_SERVER_KEY
    }));
  }

  // Notification Management
  async sendNotification(notificationData) {
    try {
      const notification = new Notification(
        notificationData.templateId,
        notificationData.recipients,
        notificationData.variables,
        notificationData.channels
      );
      
      // Set additional properties
      if (notificationData.priority) notification.priority = notificationData.priority;
      if (notificationData.scheduleAt) notification.scheduledAt = new Date(notificationData.scheduleAt);
      
      // Store notification
      this.notifications.set(notification.id, notification);
      
      // Check if scheduled
      if (notification.scheduledAt > new Date()) {
        this.emit("notificationScheduled", notification);
        return notification;
      }
      
      // Process immediately
      await this.processNotification(notification);
      
      return notification;
      
    } catch (error) {
      console.error("Notification sending error:", error);
      throw error;
    }
  }

  async processNotification(notification) {
    try {
      notification.status = "processing";
      notification.updatedAt = new Date();
      
      // Get template
      const template = this.templates.get(notification.templateId);
      if (!template) {
        throw new Error("Template not found");
      }
      
      // Process each recipient
      for (const recipient of notification.recipients) {
        await this.processRecipient(notification, recipient, template);
      }
      
      notification.status = "sent";
      notification.updatedAt = new Date();
      
      this.emit("notificationSent", notification);
      
    } catch (error) {
      notification.status = "failed";
      notification.error = error.message;
      notification.updatedAt = new Date();
      
      this.emit("notificationFailed", notification);
    }
  }

  async processRecipient(notification, recipient, template) {
    // Check user preferences
    const preferences = this.getUserPreferences(recipient.userId);
    if (!preferences) {
      console.warn(`No preferences found for user ${recipient.userId}`);
      return;
    }
    
    // Process each channel
    for (const channel of notification.channels) {
      if (!this.isChannelEnabled(preferences, channel)) {
        continue;
      }
      
      if (this.isQuietHours(preferences)) {
        // Schedule for later
        this.scheduleForLater(notification, recipient, channel);
        continue;
      }
      
      await this.sendToChannel(notification, recipient, channel, template);
    }
  }

  async sendToChannel(notification, recipient, channel, template) {
    try {
      // Generate message
      const message = this.generateMessage(template, notification.variables, channel);
      
      // Create delivery record
      const delivery = new Delivery(
        notification.id,
        recipient.userId,
        channel,
        message
      );
      
      this.deliveries.set(delivery.id, delivery);
      
      // Add to delivery queue
      this.deliveryQueue.push(delivery);
      
      this.emit("deliveryQueued", delivery);
      
    } catch (error) {
      console.error("Channel processing error:", error);
    }
  }

  // Template Management
  async createTemplate(templateData) {
    const template = new Template(
      templateData.name,
      templateData.type,
      templateData.content
    );
    
    if (templateData.subject) template.subject = templateData.subject;
    if (templateData.variables) template.variables = templateData.variables;
    if (templateData.createdBy) template.createdBy = templateData.createdBy;
    
    this.templates.set(template.id, template);
    
    this.emit("templateCreated", template);
    
    return template;
  }

  generateMessage(template, variables, channel) {
    let message = template.content;
    let subject = template.subject;
    
    // Replace variables
    Object.entries(variables).forEach(([key, value]) => {
      const placeholder = `{{${key}}}`;
      message = message.replace(new RegExp(placeholder, "g"), value);
      subject = subject.replace(new RegExp(placeholder, "g"), value);
    });
    
    return {
      subject,
      content: message,
      channel
    };
  }

  // Delivery Processing
  startDeliveryProcessor() {
    setInterval(() => {
      this.processDeliveryQueue();
    }, 1000); // Process every second
  }

  async processDeliveryQueue() {
    if (this.isProcessing || this.deliveryQueue.length === 0) {
      return;
    }
    
    this.isProcessing = true;
    
    while (this.deliveryQueue.length > 0) {
      const delivery = this.deliveryQueue.shift();
      
      try {
        await this.deliverMessage(delivery);
      } catch (error) {
        console.error("Delivery processing error:", error);
        this.handleDeliveryFailure(delivery, error);
      }
    }
    
    this.isProcessing = false;
  }

  async deliverMessage(delivery) {
    try {
      const provider = this.selectProvider(delivery.channel);
      if (!provider) {
        throw new Error(`No provider available for channel: ${delivery.channel}`);
      }
      
      // Check rate limit
      if (!this.checkRateLimit(provider)) {
        // Re-queue for later
        this.deliveryQueue.push(delivery);
        return;
      }
      
      // Send message
      const result = await this.sendViaProvider(provider, delivery);
      
      // Update delivery status
      delivery.status = "sent";
      delivery.sentAt = new Date();
      delivery.providerId = provider.name;
      delivery.providerResponse = result;
      
      this.emit("deliverySent", delivery);
      
      // Simulate delivery confirmation (in production, this would come from provider)
      setTimeout(() => {
        this.confirmDelivery(delivery);
      }, Math.random() * 5000);
      
    } catch (error) {
      this.handleDeliveryFailure(delivery, error);
    }
  }

  async sendViaProvider(provider, delivery) {
    // Simulate provider API call
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        // Simulate 95% success rate
        if (Math.random() > 0.05) {
          resolve({
            messageId: `msg_${Date.now()}`,
            status: "sent",
            timestamp: new Date()
          });
        } else {
          reject(new Error("Provider delivery failed"));
        }
      }, 1000);
    });
  }

  confirmDelivery(delivery) {
    delivery.status = "delivered";
    delivery.deliveredAt = new Date();
    
    this.emit("deliveryDelivered", delivery);
    
    // Update notification stats
    this.updateNotificationStats(delivery);
  }

  handleDeliveryFailure(delivery, error) {
    delivery.retryCount++;
    delivery.error = error.message;
    
    if (delivery.retryCount < delivery.maxRetries) {
      // Retry with exponential backoff
      const delay = Math.pow(2, delivery.retryCount) * 1000;
      setTimeout(() => {
        this.deliveryQueue.push(delivery);
      }, delay);
      
      this.emit("deliveryRetry", delivery);
    } else {
      delivery.status = "failed";
      delivery.failedAt = new Date();
      
      this.emit("deliveryFailed", delivery);
    }
  }

  // User Preferences
  getUserPreferences(userId) {
    return this.userPreferences.get(userId) || new UserPreferences(userId);
  }

  async updateUserPreferences(userId, preferences) {
    const userPrefs = this.getUserPreferences(userId);
    
    if (preferences.channels) {
      userPrefs.channels = { ...userPrefs.channels, ...preferences.channels };
    }
    
    if (preferences.categories) {
      userPrefs.categories = { ...userPrefs.categories, ...preferences.categories };
    }
    
    if (preferences.quietHours) {
      userPrefs.quietHours = { ...userPrefs.quietHours, ...preferences.quietHours };
    }
    
    userPrefs.updatedAt = new Date();
    this.userPreferences.set(userId, userPrefs);
    
    this.emit("preferencesUpdated", { userId, preferences: userPrefs });
    
    return userPrefs;
  }

  isChannelEnabled(preferences, channel) {
    return preferences.channels[channel]?.enabled === true;
  }

  isQuietHours(preferences) {
    if (!preferences.quietHours.enabled) {
      return false;
    }
    
    const now = new Date();
    const currentTime = now.toTimeString().substr(0, 5);
    const { start, end } = preferences.quietHours;
    
    if (start < end) {
      return currentTime >= start && currentTime <= end;
    } else {
      return currentTime >= start || currentTime <= end;
    }
  }

  // Provider Management
  selectProvider(channel) {
    const channelProviders = Array.from(this.providers.values())
      .filter(provider => provider.type === channel && provider.isActive);
    
    if (channelProviders.length === 0) {
      return null;
    }
    
    // Select provider with least recent usage
    return channelProviders.reduce((selected, provider) => {
      if (!selected || !provider.lastUsed || provider.lastUsed < selected.lastUsed) {
        return provider;
      }
      return selected;
    });
  }

  checkRateLimit(provider) {
    const now = Date.now();
    const windowStart = now - provider.rateLimit.window;
    
    // Simplified rate limiting (in production, use Redis)
    if (!provider.lastUsed || provider.lastUsed < windowStart) {
      provider.lastUsed = now;
      return true;
    }
    
    return false;
  }

  // Analytics
  updateNotificationStats(delivery) {
    const notification = this.notifications.get(delivery.notificationId);
    if (!notification) return;
    
    if (!notification.deliveryStats[delivery.channel]) {
      notification.deliveryStats[delivery.channel] = {
        sent: 0,
        delivered: 0,
        failed: 0,
        pending: 0
      };
    }
    
    const stats = notification.deliveryStats[delivery.channel];
    
    if (delivery.status === "delivered") {
      stats.delivered++;
    } else if (delivery.status === "failed") {
      stats.failed++;
    } else if (delivery.status === "sent") {
      stats.sent++;
    } else {
      stats.pending++;
    }
    
    // Check if all deliveries are complete
    const totalDeliveries = Object.values(stats).reduce((sum, count) => sum + count, 0);
    const completedDeliveries = stats.delivered + stats.failed;
    
    if (completedDeliveries === totalDeliveries) {
      notification.status = "delivered";
      notification.completedAt = new Date();
      notification.updatedAt = new Date();
      
      this.emit("notificationCompleted", notification);
    }
  }

  // Scheduling
  startScheduler() {
    setInterval(() => {
      this.processScheduledNotifications();
    }, 60000); // Check every minute
  }

  processScheduledNotifications() {
    const now = new Date();
    
    for (const notification of this.notifications.values()) {
      if (notification.status === "queued" && 
          notification.scheduledAt <= now) {
        this.processNotification(notification);
      }
    }
  }

  // Background Tasks
  startCleanupTask() {
    setInterval(() => {
      this.cleanupOldNotifications();
    }, 86400000); // Run daily
  }

  cleanupOldNotifications() {
    const cutoff = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000); // 30 days ago
    const oldNotifications = [];
    
    for (const [id, notification] of this.notifications) {
      if (notification.createdAt < cutoff) {
        oldNotifications.push(id);
      }
    }
    
    oldNotifications.forEach(id => {
      this.notifications.delete(id);
    });
    
    if (oldNotifications.length > 0) {
      this.emit("notificationsCleaned", oldNotifications.length);
    }
  }

  // Utility Methods
  generateID() {
    return uuidv4();
  }
}
```

### Express.js API Implementation

```javascript
const express = require("express");
const cors = require("cors");
const { NotificationService } = require("./services/NotificationService");

class NotificationAPI {
  constructor() {
    this.app = express();
    this.notificationService = new NotificationService();
    
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
    // Notification management
    this.app.post("/api/notifications/send", this.sendNotification.bind(this));
    this.app.post("/api/notifications/bulk", this.sendBulkNotification.bind(this));
    this.app.get("/api/notifications/:notificationId", this.getNotification.bind(this));
    this.app.get("/api/notifications", this.getNotifications.bind(this));
    this.app.put("/api/notifications/:notificationId/cancel", this.cancelNotification.bind(this));
    
    // Templates
    this.app.post("/api/templates", this.createTemplate.bind(this));
    this.app.get("/api/templates", this.getTemplates.bind(this));
    this.app.put("/api/templates/:templateId", this.updateTemplate.bind(this));
    this.app.delete("/api/templates/:templateId", this.deleteTemplate.bind(this));
    
    // Scheduling
    this.app.post("/api/notifications/schedule", this.scheduleNotification.bind(this));
    this.app.get("/api/notifications/scheduled", this.getScheduledNotifications.bind(this));
    
    // User preferences
    this.app.get("/api/users/:userId/preferences", this.getUserPreferences.bind(this));
    this.app.put("/api/users/:userId/preferences", this.updateUserPreferences.bind(this));
    this.app.post("/api/users/:userId/opt-out", this.optOutUser.bind(this));
    
    // Analytics
    this.app.get("/api/notifications/:notificationId/status", this.getNotificationStatus.bind(this));
    this.app.get("/api/analytics/delivery", this.getDeliveryAnalytics.bind(this));
    this.app.get("/api/analytics/channels", this.getChannelAnalytics.bind(this));
    
    // Health check
    this.app.get("/health", (req, res) => {
      res.json({
        status: "healthy",
        timestamp: new Date(),
        totalNotifications: this.notificationService.notifications.size,
        totalTemplates: this.notificationService.templates.size,
        queueSize: this.notificationService.deliveryQueue.length
      });
    });
  }

  setupEventHandlers() {
    this.notificationService.on("notificationSent", (notification) => {
      console.log(`Notification sent: ${notification.id}`);
    });
    
    this.notificationService.on("deliveryDelivered", (delivery) => {
      console.log(`Delivery confirmed: ${delivery.id}`);
    });
    
    this.notificationService.on("deliveryFailed", (delivery) => {
      console.log(`Delivery failed: ${delivery.id} - ${delivery.error}`);
    });
  }

  // HTTP Handlers
  async sendNotification(req, res) {
    try {
      const notification = await this.notificationService.sendNotification(req.body);
      
      res.status(201).json({
        success: true,
        data: {
          notificationId: notification.id,
          status: notification.status,
          recipients: notification.recipients.length,
          channels: notification.channels,
          scheduledAt: notification.scheduledAt,
          createdAt: notification.createdAt
        }
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async getNotification(req, res) {
    try {
      const { notificationId } = req.params;
      const notification = this.notificationService.notifications.get(notificationId);
      
      if (!notification) {
        return res.status(404).json({ error: "Notification not found" });
      }
      
      res.json({
        success: true,
        data: {
          notificationId: notification.id,
          status: notification.status,
          totalRecipients: notification.recipients.length,
          deliveryStats: notification.deliveryStats,
          createdAt: notification.createdAt,
          completedAt: notification.completedAt,
          error: notification.error
        }
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async createTemplate(req, res) {
    try {
      const template = await this.notificationService.createTemplate(req.body);
      
      res.status(201).json({
        success: true,
        data: {
          templateId: template.id,
          name: template.name,
          type: template.type,
          isActive: template.isActive,
          createdAt: template.createdAt
        }
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async getUserPreferences(req, res) {
    try {
      const { userId } = req.params;
      const preferences = this.notificationService.getUserPreferences(userId);
      
      res.json({
        success: true,
        data: preferences
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async updateUserPreferences(req, res) {
    try {
      const { userId } = req.params;
      const preferences = await this.notificationService.updateUserPreferences(userId, req.body);
      
      res.json({
        success: true,
        data: preferences
      });
    } catch (error) {
      res.status(400).json({ 
        success: false,
        error: error.message 
      });
    }
  }

  async getDeliveryAnalytics(req, res) {
    try {
      const { startDate, endDate, channel } = req.query;
      
      const deliveries = Array.from(this.notificationService.deliveries.values());
      
      let filteredDeliveries = deliveries;
      
      if (startDate) {
        filteredDeliveries = filteredDeliveries.filter(d => d.sentAt >= new Date(startDate));
      }
      
      if (endDate) {
        filteredDeliveries = filteredDeliveries.filter(d => d.sentAt <= new Date(endDate));
      }
      
      if (channel) {
        filteredDeliveries = filteredDeliveries.filter(d => d.channel === channel);
      }
      
      const analytics = {
        total: filteredDeliveries.length,
        byStatus: this.groupBy(filteredDeliveries, "status"),
        byChannel: this.groupBy(filteredDeliveries, "channel"),
        successRate: this.calculateSuccessRate(filteredDeliveries)
      };
      
      res.json({
        success: true,
        data: analytics
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  // Utility Methods
  groupBy(array, key) {
    return array.reduce((groups, item) => {
      const group = item[key];
      groups[group] = groups[group] || [];
      groups[group].push(item);
      return groups;
    }, {});
  }

  calculateSuccessRate(deliveries) {
    const total = deliveries.length;
    const successful = deliveries.filter(d => d.status === "delivered").length;
    return total > 0 ? (successful / total) * 100 : 0;
  }

  start(port = 3000) {
    this.app.listen(port, () => {
      console.log(`Notification API server running on port ${port}`);
    });
  }
}

// Start server
if (require.main === module) {
  const api = new NotificationAPI();
  api.start(3000);
}

module.exports = { NotificationAPI };
```

## Key Features

### Multi-Channel Support
- **Email**: SMTP and SendGrid integration
- **SMS**: Twilio and other SMS providers
- **Push Notifications**: Firebase Cloud Messaging
- **Webhooks**: Custom webhook delivery

### Template Management
- **Dynamic Templates**: Variable substitution
- **Multi-Channel Templates**: Different content per channel
- **Template Versioning**: Template history and rollback
- **Rich Content**: HTML, plain text, and structured data

### Delivery Management
- **Queue System**: Reliable message delivery
- **Retry Logic**: Exponential backoff for failures
- **Rate Limiting**: Provider-specific rate limits
- **Delivery Tracking**: Real-time status updates

### User Experience
- **Preferences**: Channel and frequency preferences
- **Quiet Hours**: Do not disturb functionality
- **Opt-out Management**: Easy unsubscribe options
- **Bulk Operations**: Mass notification support

## Extension Ideas

### Advanced Features
1. **A/B Testing**: Template and timing optimization
2. **Personalization**: AI-driven content customization
3. **Rich Media**: Image and video attachments
4. **Interactive Messages**: Buttons and quick replies
5. **Delivery Optimization**: Smart timing and channel selection

### Enterprise Features
1. **Multi-tenancy**: Isolated notification environments
2. **Advanced Analytics**: Detailed delivery insights
3. **Compliance**: GDPR and CAN-SPAM compliance
4. **Integration APIs**: Third-party service integration
5. **Audit Trails**: Complete notification history
