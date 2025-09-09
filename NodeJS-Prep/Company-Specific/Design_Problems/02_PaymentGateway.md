# 02. Payment Gateway - Financial Transaction System

## Title & Summary
Design and implement a payment gateway system using Node.js that processes payments, handles refunds, manages transactions, and integrates with multiple payment providers.

## Problem Statement

Build a comprehensive payment gateway that handles:

1. **Payment Processing**: Support multiple payment methods (cards, UPI, net banking)
2. **Transaction Management**: Track payment status and lifecycle
3. **Refund Processing**: Handle full and partial refunds
4. **Webhook Integration**: Real-time payment status updates
5. **Fraud Detection**: Basic fraud prevention mechanisms
6. **Settlement**: Daily settlement and reconciliation

## Requirements & Constraints

### Functional Requirements
- Process payments with multiple methods
- Handle payment failures and retries
- Process refunds and chargebacks
- Generate transaction reports
- Real-time webhook notifications
- Merchant dashboard integration

### Non-Functional Requirements
- **Latency**: < 2 seconds for payment processing
- **Availability**: 99.9% uptime
- **Throughput**: 10,000 transactions per minute
- **Security**: PCI DSS compliance
- **Reliability**: 99.95% transaction success rate
- **Scalability**: Handle 1M+ transactions per day

## API / Interfaces

### REST Endpoints

```javascript
// Payment Processing
POST   /api/payments/process
GET    /api/payments/{paymentID}/status
POST   /api/payments/{paymentID}/capture
POST   /api/payments/{paymentID}/void

// Refund Processing
POST   /api/refunds/process
GET    /api/refunds/{refundID}/status

// Transaction Management
GET    /api/transactions
GET    /api/transactions/{transactionID}
GET    /api/transactions/merchant/{merchantID}

// Webhook Management
POST   /api/webhooks/payment
POST   /api/webhooks/refund
GET    /api/webhooks/events

// Settlement
GET    /api/settlements/daily
POST   /api/settlements/process
```

### Request/Response Examples

```json
// Process Payment
POST /api/payments/process
{
  "merchantID": "merchant_123",
  "amount": 1000,
  "currency": "INR",
  "paymentMethod": "card",
  "cardDetails": {
    "number": "4111111111111111",
    "expiryMonth": "12",
    "expiryYear": "2025",
    "cvv": "123"
  },
  "customerInfo": {
    "email": "customer@example.com",
    "phone": "+919876543210"
  },
  "orderID": "order_456"
}

// Payment Response
{
  "success": true,
  "data": {
    "paymentID": "pay_789",
    "status": "processing",
    "amount": 1000,
    "currency": "INR",
    "transactionID": "txn_abc123",
    "createdAt": "2024-01-15T10:30:00Z"
  }
}

// Webhook Payload
{
  "event": "payment.completed",
  "data": {
    "paymentID": "pay_789",
    "status": "completed",
    "amount": 1000,
    "transactionID": "txn_abc123",
    "timestamp": "2024-01-15T10:30:05Z"
  }
}
```

## Data Model

### Core Entities

```javascript
// Payment Entity
class Payment {
    constructor(merchantID, amount, currency, paymentMethod) {
        this.id = this.generateID();
        this.merchantID = merchantID;
        this.amount = amount;
        this.currency = currency;
        this.paymentMethod = paymentMethod;
        this.status = 'pending';
        this.orderID = null;
        this.customerInfo = {};
        this.paymentDetails = {};
        this.gatewayTransactionID = null;
        this.gatewayResponse = null;
        this.createdAt = new Date();
        this.updatedAt = new Date();
        this.processedAt = null;
        this.failedAt = null;
        this.failureReason = null;
    }
}

// Transaction Entity
class Transaction {
    constructor(paymentID, type, amount, status) {
        this.id = this.generateID();
        this.paymentID = paymentID;
        this.type = type; // 'payment', 'refund', 'chargeback'
        this.amount = amount;
        this.status = status;
        this.gatewayTransactionID = null;
        this.gatewayResponse = null;
        this.createdAt = new Date();
        this.updatedAt = new Date();
    }
}

// Refund Entity
class Refund {
    constructor(paymentID, amount, reason) {
        this.id = this.generateID();
        this.paymentID = paymentID;
        this.amount = amount;
        this.reason = reason;
        this.status = 'pending';
        this.gatewayRefundID = null;
        this.gatewayResponse = null;
        this.createdAt = new Date();
        this.updatedAt = new Date();
    }
}

// Settlement Entity
class Settlement {
    constructor(merchantID, date, totalAmount, transactionCount) {
        this.id = this.generateID();
        this.merchantID = merchantID;
        this.date = date;
        this.totalAmount = totalAmount;
        this.transactionCount = transactionCount;
        this.status = 'pending';
        this.processedAt = null;
        this.createdAt = new Date();
    }
}
```

## Approach Overview

### Simple Solution (MVP)
1. In-memory storage with basic validation
2. Single payment provider integration
3. Simple status tracking
4. Basic error handling

### Production-Ready Design
1. **Modular Architecture**: Separate payment providers
2. **Event-Driven**: Use EventEmitter for payment events
3. **Persistence Layer**: Database for transaction history
4. **Retry Mechanism**: Exponential backoff for failures
5. **Fraud Detection**: Basic rule-based fraud prevention
6. **Webhook System**: Reliable webhook delivery

## Detailed Design

### Core Service Implementation

```javascript
const EventEmitter = require('events');
const crypto = require('crypto');
const { v4: uuidv4 } = require('uuid');

class PaymentGatewayService extends EventEmitter {
    constructor() {
        super();
        this.payments = new Map();
        this.transactions = new Map();
        this.refunds = new Map();
        this.settlements = new Map();
        this.webhooks = new Map();
        
        // Payment providers
        this.providers = new Map();
        this.initializeProviders();
        
        // Fraud detection
        this.fraudDetector = new FraudDetectionService();
        
        // Retry mechanism
        this.retryQueue = [];
        this.startRetryProcessor();
    }
    
    initializeProviders() {
        this.providers.set('card', new CardPaymentProvider());
        this.providers.set('upi', new UPIPaymentProvider());
        this.providers.set('netbanking', new NetBankingProvider());
    }
    
    // Payment Processing
    async processPayment(paymentData) {
        try {
            // Validate payment data
            this.validatePaymentData(paymentData);
            
            // Create payment record
            const payment = new Payment(
                paymentData.merchantID,
                paymentData.amount,
                paymentData.currency,
                paymentData.paymentMethod
            );
            
            payment.orderID = paymentData.orderID;
            payment.customerInfo = paymentData.customerInfo;
            payment.paymentDetails = paymentData.paymentDetails;
            
            // Store payment
            this.payments.set(payment.id, payment);
            
            // Fraud detection
            const fraudCheck = await this.fraudDetector.checkFraud(payment);
            if (fraudCheck.isFraudulent) {
                payment.status = 'failed';
                payment.failureReason = 'Fraud detected';
                payment.failedAt = new Date();
                
                this.emit('paymentFailed', payment);
                return payment;
            }
            
            // Process with payment provider
            const provider = this.providers.get(payment.paymentMethod);
            if (!provider) {
                throw new Error('Unsupported payment method');
            }
            
            const result = await provider.processPayment(payment);
            
            // Update payment status
            payment.status = result.status;
            payment.gatewayTransactionID = result.transactionID;
            payment.gatewayResponse = result.response;
            payment.updatedAt = new Date();
            
            if (result.status === 'completed') {
                payment.processedAt = new Date();
                this.emit('paymentCompleted', payment);
            } else if (result.status === 'failed') {
                payment.failedAt = new Date();
                payment.failureReason = result.error;
                this.emit('paymentFailed', payment);
            }
            
            // Create transaction record
            const transaction = new Transaction(
                payment.id,
                'payment',
                payment.amount,
                payment.status
            );
            transaction.gatewayTransactionID = result.transactionID;
            transaction.gatewayResponse = result.response;
            
            this.transactions.set(transaction.id, transaction);
            
            // Send webhook
            await this.sendWebhook(payment);
            
            return payment;
            
        } catch (error) {
            console.error('Payment processing error:', error);
            throw error;
        }
    }
    
    // Refund Processing
    async processRefund(paymentID, refundData) {
        try {
            const payment = this.payments.get(paymentID);
            if (!payment) {
                throw new Error('Payment not found');
            }
            
            if (payment.status !== 'completed') {
                throw new Error('Can only refund completed payments');
            }
            
            if (refundData.amount > payment.amount) {
                throw new Error('Refund amount cannot exceed payment amount');
            }
            
            // Create refund record
            const refund = new Refund(
                paymentID,
                refundData.amount,
                refundData.reason
            );
            
            this.refunds.set(refund.id, refund);
            
            // Process with payment provider
            const provider = this.providers.get(payment.paymentMethod);
            const result = await provider.processRefund(payment, refund);
            
            // Update refund status
            refund.status = result.status;
            refund.gatewayRefundID = result.refundID;
            refund.gatewayResponse = result.response;
            refund.updatedAt = new Date();
            
            // Create transaction record
            const transaction = new Transaction(
                paymentID,
                'refund',
                refund.amount,
                refund.status
            );
            transaction.gatewayTransactionID = result.refundID;
            transaction.gatewayResponse = result.response;
            
            this.transactions.set(transaction.id, transaction);
            
            // Send webhook
            await this.sendRefundWebhook(refund);
            
            return refund;
            
        } catch (error) {
            console.error('Refund processing error:', error);
            throw error;
        }
    }
    
    // Webhook Management
    async sendWebhook(payment) {
        const webhookData = {
            event: `payment.${payment.status}`,
            data: {
                paymentID: payment.id,
                status: payment.status,
                amount: payment.amount,
                currency: payment.currency,
                transactionID: payment.gatewayTransactionID,
                timestamp: payment.updatedAt
            }
        };
        
        // Store webhook for retry mechanism
        const webhook = {
            id: uuidv4(),
            url: this.getMerchantWebhookURL(payment.merchantID),
            payload: webhookData,
            attempts: 0,
            maxAttempts: 3,
            nextRetryAt: new Date(),
            status: 'pending'
        };
        
        this.webhooks.set(webhook.id, webhook);
        
        // Send webhook
        await this.deliverWebhook(webhook);
    }
    
    async deliverWebhook(webhook) {
        try {
            const response = await fetch(webhook.url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Webhook-Signature': this.generateWebhookSignature(webhook.payload)
                },
                body: JSON.stringify(webhook.payload)
            });
            
            if (response.ok) {
                webhook.status = 'delivered';
                webhook.deliveredAt = new Date();
            } else {
                throw new Error(`Webhook delivery failed: ${response.status}`);
            }
            
        } catch (error) {
            webhook.attempts++;
            webhook.status = 'failed';
            
            if (webhook.attempts < webhook.maxAttempts) {
                // Schedule retry with exponential backoff
                const delay = Math.pow(2, webhook.attempts) * 1000;
                webhook.nextRetryAt = new Date(Date.now() + delay);
                webhook.status = 'pending';
                
                this.retryQueue.push(webhook);
            }
            
            console.error('Webhook delivery failed:', error);
        }
    }
    
    // Retry Mechanism
    startRetryProcessor() {
        setInterval(() => {
            this.processRetryQueue();
        }, 5000); // Check every 5 seconds
    }
    
    processRetryQueue() {
        const now = new Date();
        const readyWebhooks = this.retryQueue.filter(webhook => 
            webhook.nextRetryAt <= now
        );
        
        readyWebhooks.forEach(webhook => {
            this.deliverWebhook(webhook);
            const index = this.retryQueue.indexOf(webhook);
            if (index > -1) {
                this.retryQueue.splice(index, 1);
            }
        });
    }
    
    // Settlement Processing
    async processDailySettlement(merchantID, date) {
        try {
            const transactions = this.getTransactionsForSettlement(merchantID, date);
            
            const totalAmount = transactions.reduce((sum, tx) => {
                return tx.type === 'payment' ? sum + tx.amount : sum - tx.amount;
            }, 0);
            
            const settlement = new Settlement(
                merchantID,
                date,
                totalAmount,
                transactions.length
            );
            
            this.settlements.set(settlement.id, settlement);
            
            // Process settlement with payment provider
            const provider = this.providers.get('card'); // Default provider
            await provider.processSettlement(settlement);
            
            settlement.status = 'completed';
            settlement.processedAt = new Date();
            
            this.emit('settlementCompleted', settlement);
            
            return settlement;
            
        } catch (error) {
            console.error('Settlement processing error:', error);
            throw error;
        }
    }
    
    // Utility Methods
    validatePaymentData(paymentData) {
        if (!paymentData.merchantID || !paymentData.amount || !paymentData.currency) {
            throw new Error('Missing required payment data');
        }
        
        if (paymentData.amount <= 0) {
            throw new Error('Invalid payment amount');
        }
        
        if (!['INR', 'USD', 'EUR'].includes(paymentData.currency)) {
            throw new Error('Unsupported currency');
        }
    }
    
    generateWebhookSignature(payload) {
        const secret = process.env.WEBHOOK_SECRET || 'default-secret';
        return crypto
            .createHmac('sha256', secret)
            .update(JSON.stringify(payload))
            .digest('hex');
    }
    
    getMerchantWebhookURL(merchantID) {
        // In production, fetch from database
        return `https://merchant-${merchantID}.example.com/webhooks/payment`;
    }
    
    getTransactionsForSettlement(merchantID, date) {
        return Array.from(this.transactions.values()).filter(tx => {
            const payment = this.payments.get(tx.paymentID);
            return payment && 
                   payment.merchantID === merchantID &&
                   tx.createdAt.toDateString() === date.toDateString();
        });
    }
    
    generateID() {
        return uuidv4();
    }
}
```

### Payment Provider Implementation

```javascript
// Base Payment Provider
class PaymentProvider {
    constructor(name) {
        this.name = name;
        this.baseURL = process.env[`${name.toUpperCase()}_API_URL`];
        this.apiKey = process.env[`${name.toUpperCase()}_API_KEY`];
    }
    
    async processPayment(payment) {
        throw new Error('processPayment must be implemented');
    }
    
    async processRefund(payment, refund) {
        throw new Error('processRefund must be implemented');
    }
    
    async processSettlement(settlement) {
        throw new Error('processSettlement must be implemented');
    }
}

// Card Payment Provider
class CardPaymentProvider extends PaymentProvider {
    constructor() {
        super('card');
    }
    
    async processPayment(payment) {
        try {
            // Simulate API call to card processor
            const response = await this.callCardAPI('charge', {
                amount: payment.amount,
                currency: payment.currency,
                card: payment.paymentDetails,
                order_id: payment.orderID
            });
            
            return {
                status: response.status === 'success' ? 'completed' : 'failed',
                transactionID: response.transaction_id,
                response: response
            };
            
        } catch (error) {
            return {
                status: 'failed',
                transactionID: null,
                response: { error: error.message }
            };
        }
    }
    
    async processRefund(payment, refund) {
        try {
            const response = await this.callCardAPI('refund', {
                transaction_id: payment.gatewayTransactionID,
                amount: refund.amount,
                reason: refund.reason
            });
            
            return {
                status: response.status === 'success' ? 'completed' : 'failed',
                refundID: response.refund_id,
                response: response
            };
            
        } catch (error) {
            return {
                status: 'failed',
                refundID: null,
                response: { error: error.message }
            };
        }
    }
    
    async callCardAPI(endpoint, data) {
        // Simulate API call
        return new Promise((resolve) => {
            setTimeout(() => {
                // Simulate 95% success rate
                const success = Math.random() > 0.05;
                
                if (success) {
                    resolve({
                        status: 'success',
                        transaction_id: `txn_${Date.now()}`,
                        refund_id: `ref_${Date.now()}`,
                        amount: data.amount,
                        currency: data.currency
                    });
                } else {
                    resolve({
                        status: 'failed',
                        error: 'Payment declined by bank'
                    });
                }
            }, 1000); // Simulate network delay
        });
    }
}

// UPI Payment Provider
class UPIPaymentProvider extends PaymentProvider {
    constructor() {
        super('upi');
    }
    
    async processPayment(payment) {
        try {
            const response = await this.callUPIAPI('collect', {
                amount: payment.amount,
                vpa: payment.paymentDetails.vpa,
                order_id: payment.orderID
            });
            
            return {
                status: response.status === 'success' ? 'completed' : 'failed',
                transactionID: response.upi_transaction_id,
                response: response
            };
            
        } catch (error) {
            return {
                status: 'failed',
                transactionID: null,
                response: { error: error.message }
            };
        }
    }
    
    async callUPIAPI(endpoint, data) {
        // Simulate UPI API call
        return new Promise((resolve) => {
            setTimeout(() => {
                const success = Math.random() > 0.02; // 98% success rate for UPI
                
                if (success) {
                    resolve({
                        status: 'success',
                        upi_transaction_id: `upi_${Date.now()}`,
                        amount: data.amount
                    });
                } else {
                    resolve({
                        status: 'failed',
                        error: 'UPI transaction failed'
                    });
                }
            }, 800);
        });
    }
}
```

### Fraud Detection Service

```javascript
class FraudDetectionService {
    constructor() {
        this.rules = [
            new AmountLimitRule(),
            new VelocityRule(),
            new BlacklistRule(),
            new GeolocationRule()
        ];
    }
    
    async checkFraud(payment) {
        const fraudScore = 0;
        const reasons = [];
        
        for (const rule of this.rules) {
            const result = await rule.evaluate(payment);
            if (result.isFraudulent) {
                return {
                    isFraudulent: true,
                    score: 100,
                    reasons: [result.reason]
                };
            }
            fraudScore += result.score;
            if (result.reason) {
                reasons.push(result.reason);
            }
        }
        
        return {
            isFraudulent: fraudScore > 80,
            score: fraudScore,
            reasons: reasons
        };
    }
}

// Fraud Detection Rules
class AmountLimitRule {
    async evaluate(payment) {
        if (payment.amount > 100000) { // 1 lakh limit
            return {
                isFraudulent: true,
                score: 100,
                reason: 'Amount exceeds limit'
            };
        }
        return { isFraudulent: false, score: 0 };
    }
}

class VelocityRule {
    async evaluate(payment) {
        // Check if too many transactions in short time
        // This would require access to transaction history
        return { isFraudulent: false, score: 0 };
    }
}
```

## Express.js API Implementation

```javascript
const express = require('express');
const cors = require('cors');
const { PaymentGatewayService } = require('./services/PaymentGatewayService');

class PaymentGatewayAPI {
    constructor() {
        this.app = express();
        this.paymentService = new PaymentGatewayService();
        
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
        
        // Rate limiting
        this.app.use(this.rateLimitMiddleware());
    }
    
    setupRoutes() {
        // Payment routes
        this.app.post('/api/payments/process', this.processPayment.bind(this));
        this.app.get('/api/payments/:paymentID/status', this.getPaymentStatus.bind(this));
        this.app.post('/api/payments/:paymentID/capture', this.capturePayment.bind(this));
        
        // Refund routes
        this.app.post('/api/refunds/process', this.processRefund.bind(this));
        this.app.get('/api/refunds/:refundID/status', this.getRefundStatus.bind(this));
        
        // Transaction routes
        this.app.get('/api/transactions', this.getTransactions.bind(this));
        this.app.get('/api/transactions/:transactionID', this.getTransaction.bind(this));
        
        // Webhook routes
        this.app.post('/api/webhooks/payment', this.handlePaymentWebhook.bind(this));
        this.app.post('/api/webhooks/refund', this.handleRefundWebhook.bind(this));
        
        // Settlement routes
        this.app.get('/api/settlements/daily', this.getDailySettlements.bind(this));
        this.app.post('/api/settlements/process', this.processSettlement.bind(this));
        
        // Health check
        this.app.get('/health', (req, res) => {
            res.json({ 
                status: 'healthy', 
                timestamp: new Date(),
                totalPayments: this.paymentService.payments.size,
                totalTransactions: this.paymentService.transactions.size
            });
        });
    }
    
    setupEventHandlers() {
        this.paymentService.on('paymentCompleted', (payment) => {
            console.log(`Payment completed: ${payment.id}`);
        });
        
        this.paymentService.on('paymentFailed', (payment) => {
            console.log(`Payment failed: ${payment.id} - ${payment.failureReason}`);
        });
        
        this.paymentService.on('settlementCompleted', (settlement) => {
            console.log(`Settlement completed: ${settlement.id}`);
        });
    }
    
    // HTTP Handlers
    async processPayment(req, res) {
        try {
            const payment = await this.paymentService.processPayment(req.body);
            
            res.status(201).json({
                success: true,
                data: {
                    paymentID: payment.id,
                    status: payment.status,
                    amount: payment.amount,
                    currency: payment.currency,
                    transactionID: payment.gatewayTransactionID,
                    createdAt: payment.createdAt
                }
            });
        } catch (error) {
            res.status(400).json({ 
                success: false,
                error: error.message 
            });
        }
    }
    
    async getPaymentStatus(req, res) {
        try {
            const { paymentID } = req.params;
            const payment = this.paymentService.payments.get(paymentID);
            
            if (!payment) {
                return res.status(404).json({ error: 'Payment not found' });
            }
            
            res.json({
                success: true,
                data: {
                    paymentID: payment.id,
                    status: payment.status,
                    amount: payment.amount,
                    currency: payment.currency,
                    transactionID: payment.gatewayTransactionID,
                    createdAt: payment.createdAt,
                    updatedAt: payment.updatedAt
                }
            });
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    }
    
    async processRefund(req, res) {
        try {
            const { paymentID, amount, reason } = req.body;
            
            if (!paymentID || !amount || !reason) {
                return res.status(400).json({ error: 'Missing required fields' });
            }
            
            const refund = await this.paymentService.processRefund(paymentID, {
                amount,
                reason
            });
            
            res.status(201).json({
                success: true,
                data: {
                    refundID: refund.id,
                    paymentID: refund.paymentID,
                    amount: refund.amount,
                    status: refund.status,
                    createdAt: refund.createdAt
                }
            });
        } catch (error) {
            res.status(400).json({ 
                success: false,
                error: error.message 
            });
        }
    }
    
    async getTransactions(req, res) {
        try {
            const { merchantID, limit = 50, offset = 0 } = req.query;
            
            let transactions = Array.from(this.paymentService.transactions.values());
            
            if (merchantID) {
                transactions = transactions.filter(tx => {
                    const payment = this.paymentService.payments.get(tx.paymentID);
                    return payment && payment.merchantID === merchantID;
                });
            }
            
            // Apply pagination
            const paginatedTransactions = transactions.slice(
                parseInt(offset), 
                parseInt(offset) + parseInt(limit)
            );
            
            res.json({
                success: true,
                data: paginatedTransactions,
                pagination: {
                    limit: parseInt(limit),
                    offset: parseInt(offset),
                    total: transactions.length
                }
            });
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    }
    
    async handlePaymentWebhook(req, res) {
        try {
            const signature = req.headers['x-webhook-signature'];
            const payload = req.body;
            
            // Verify webhook signature
            if (!this.verifyWebhookSignature(payload, signature)) {
                return res.status(401).json({ error: 'Invalid signature' });
            }
            
            // Process webhook
            console.log('Received payment webhook:', payload);
            
            res.json({ success: true, message: 'Webhook processed' });
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    }
    
    async processSettlement(req, res) {
        try {
            const { merchantID, date } = req.body;
            
            if (!merchantID || !date) {
                return res.status(400).json({ error: 'Missing required fields' });
            }
            
            const settlement = await this.paymentService.processDailySettlement(
                merchantID, 
                new Date(date)
            );
            
            res.status(201).json({
                success: true,
                data: {
                    settlementID: settlement.id,
                    merchantID: settlement.merchantID,
                    date: settlement.date,
                    totalAmount: settlement.totalAmount,
                    transactionCount: settlement.transactionCount,
                    status: settlement.status
                }
            });
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    }
    
    // Middleware
    rateLimitMiddleware() {
        const requests = new Map();
        const windowMs = 60000; // 1 minute
        const maxRequests = 100;
        
        return (req, res, next) => {
            const clientIP = req.ip;
            const now = Date.now();
            
            if (!requests.has(clientIP)) {
                requests.set(clientIP, { count: 1, resetTime: now + windowMs });
                return next();
            }
            
            const clientData = requests.get(clientIP);
            
            if (now > clientData.resetTime) {
                clientData.count = 1;
                clientData.resetTime = now + windowMs;
                return next();
            }
            
            if (clientData.count >= maxRequests) {
                return res.status(429).json({ error: 'Rate limit exceeded' });
            }
            
            clientData.count++;
            next();
        };
    }
    
    verifyWebhookSignature(payload, signature) {
        const expectedSignature = this.paymentService.generateWebhookSignature(payload);
        return signature === expectedSignature;
    }
    
    start(port = 3000) {
        this.app.listen(port, () => {
            console.log(`Payment Gateway API server running on port ${port}`);
        });
    }
}

// Start server
if (require.main === module) {
    const api = new PaymentGatewayAPI();
    api.start(3000);
}

module.exports = { PaymentGatewayAPI };
```

## Key Features

### Payment Processing
- Multiple payment methods support
- Real-time payment status tracking
- Comprehensive error handling
- Fraud detection integration

### Transaction Management
- Complete transaction lifecycle
- Refund and chargeback handling
- Settlement processing
- Webhook notifications

### Security & Compliance
- PCI DSS compliance considerations
- Webhook signature verification
- Rate limiting and fraud detection
- Secure data handling

### Scalability & Reliability
- Event-driven architecture
- Retry mechanisms for failed operations
- Comprehensive logging and monitoring
- Modular provider system

## Extension Ideas

### Advanced Features
1. **Multi-currency Support**: Currency conversion and international payments
2. **Subscription Payments**: Recurring payment processing
3. **Split Payments**: Multiple merchant payment splitting
4. **Payment Links**: Generate shareable payment links
5. **Mobile SDKs**: Native mobile app integration

### Enterprise Features
1. **Advanced Fraud Detection**: Machine learning-based fraud prevention
2. **Compliance Tools**: PCI DSS compliance automation
3. **Analytics Dashboard**: Real-time payment analytics
4. **Multi-tenant Support**: White-label payment solutions
5. **API Versioning**: Backward compatibility management
