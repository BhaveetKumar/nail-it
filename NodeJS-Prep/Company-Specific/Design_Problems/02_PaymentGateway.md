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

## 💬 **Discussion Points**

### **Transaction Processing & Reliability**

**Q: How do you ensure transaction consistency and prevent double spending?**
**A:**
- **Idempotency Keys**: Use unique idempotency keys for each transaction request
- **Database Transactions**: Use ACID transactions for payment processing
- **Distributed Locks**: Implement Redis-based distributed locks for critical operations
- **State Machine**: Use finite state machine for transaction lifecycle management
- **Compensation Patterns**: Implement saga pattern for distributed transaction rollback
- **Audit Trails**: Maintain complete audit logs for all transaction operations

**Q: How do you handle payment provider failures and timeouts?**
**A:**
- **Circuit Breaker Pattern**: Implement circuit breaker to prevent cascade failures
- **Retry Logic**: Exponential backoff retry with jitter for failed requests
- **Fallback Providers**: Route to alternative payment providers when primary fails
- **Timeout Management**: Set appropriate timeouts and implement graceful degradation
- **Health Checks**: Monitor provider health and route traffic accordingly
- **Dead Letter Queue**: Handle permanently failed transactions

### **Security & Compliance**

**Q: How do you protect sensitive payment data (PCI DSS compliance)?**
**A:**
- **Data Encryption**: Encrypt sensitive data at rest and in transit (AES-256)
- **Tokenization**: Replace sensitive data with tokens for storage
- **Key Management**: Use HSM (Hardware Security Module) for key storage
- **Access Controls**: Implement role-based access control and audit logging
- **Network Security**: Use VPNs and private networks for provider communication
- **Regular Audits**: Conduct regular security audits and penetration testing

**Q: How do you prevent fraud and implement risk management?**
**A:**
- **Risk Scoring**: Implement ML-based risk scoring for transactions
- **Velocity Checks**: Monitor transaction frequency and amounts per user
- **Device Fingerprinting**: Track device characteristics for fraud detection
- **Geolocation Analysis**: Flag transactions from unusual locations
- **Blacklist Management**: Maintain and check against fraud databases
- **Real-time Monitoring**: Implement real-time fraud detection and alerting

### **Scalability & Performance**

**Q: How do you handle high-volume transaction processing?**
**A:**
- **Horizontal Scaling**: Deploy multiple payment processing instances
- **Load Balancing**: Use intelligent load balancing based on provider capacity
- **Async Processing**: Implement asynchronous processing for non-critical operations
- **Database Sharding**: Partition transactions by merchant or geographic region
- **Caching Strategy**: Cache frequently accessed data (merchant configs, user data)
- **Queue Management**: Use message queues for transaction processing

**Q: How do you ensure low latency for payment processing?**
**A:**
- **Connection Pooling**: Maintain persistent connections to payment providers
- **Geographic Distribution**: Deploy processing nodes close to users
- **CDN Integration**: Use CDN for static content and API responses
- **Database Optimization**: Optimize database queries and use read replicas
- **Caching Layers**: Implement multi-level caching (Redis, application cache)
- **Provider Optimization**: Choose providers with low latency and high availability

## ❓ **Follow-up Questions**

### **Advanced Payment Features**

**Q1: How do you implement recurring payments and subscriptions?**
**A:**
```javascript
// Recurring Payment Service
class RecurringPaymentService {
  async createSubscription(subscriptionData) {
    const subscription = {
      id: this.generateID(),
      customerId: subscriptionData.customerId,
      merchantId: subscriptionData.merchantId,
      amount: subscriptionData.amount,
      frequency: subscriptionData.frequency, // daily, weekly, monthly, yearly
      startDate: subscriptionData.startDate,
      endDate: subscriptionData.endDate,
      status: 'active',
      nextBillingDate: this.calculateNextBillingDate(subscriptionData),
      paymentMethod: subscriptionData.paymentMethod,
      createdAt: new Date()
    };
    
    await this.database.subscriptions.insert(subscription);
    
    // Schedule first payment
    this.schedulePayment(subscription);
    
    return subscription;
  }
  
  async processRecurringPayments() {
    const dueSubscriptions = await this.database.subscriptions.findDue();
    
    for (const subscription of dueSubscriptions) {
      try {
        const payment = await this.paymentService.processPayment({
          amount: subscription.amount,
          customerId: subscription.customerId,
          merchantId: subscription.merchantId,
          paymentMethod: subscription.paymentMethod,
          subscriptionId: subscription.id
        });
        
        // Update next billing date
        subscription.nextBillingDate = this.calculateNextBillingDate(subscription);
        await this.database.subscriptions.update(subscription.id, subscription);
        
        // Send notification
        this.notificationService.sendPaymentNotification(subscription.customerId, payment);
        
      } catch (error) {
        // Handle failed recurring payment
        await this.handleFailedRecurringPayment(subscription, error);
      }
    }
  }
}
```

**Q2: How do you implement payment splitting and marketplace functionality?**
**A:**
```javascript
// Payment Splitting Service
class PaymentSplittingService {
  async processSplitPayment(paymentData, splits) {
    const totalSplitAmount = splits.reduce((sum, split) => sum + split.amount, 0);
    
    if (totalSplitAmount !== paymentData.amount) {
      throw new Error('Split amounts must equal total payment amount');
    }
    
    // Process main payment
    const mainPayment = await this.paymentService.processPayment(paymentData);
    
    // Process splits
    const splitResults = [];
    for (const split of splits) {
      const splitPayment = {
        id: this.generateID(),
        parentPaymentId: mainPayment.id,
        recipientId: split.recipientId,
        amount: split.amount,
        percentage: split.percentage,
        status: 'pending',
        createdAt: new Date()
      };
      
      // Transfer to recipient
      await this.transferToRecipient(splitPayment);
      splitResults.push(splitPayment);
    }
    
    return {
      mainPayment,
      splits: splitResults
    };
  }
  
  async transferToRecipient(splitPayment) {
    // Check recipient's payment method
    const recipient = await this.database.recipients.findById(splitPayment.recipientId);
    
    if (recipient.paymentMethod === 'bank_account') {
      // Process bank transfer
      await this.bankTransferService.transfer({
        amount: splitPayment.amount,
        bankAccount: recipient.bankAccount,
        reference: splitPayment.id
      });
    } else if (recipient.paymentMethod === 'wallet') {
      // Credit wallet
      await this.walletService.credit(recipient.walletId, splitPayment.amount);
    }
    
    splitPayment.status = 'completed';
    await this.database.splitPayments.update(splitPayment.id, splitPayment);
  }
}
```

**Q3: How do you implement international payments and currency conversion?**
**A:**
```javascript
// International Payment Service
class InternationalPaymentService {
  async processInternationalPayment(paymentData) {
    // Get exchange rate
    const exchangeRate = await this.exchangeRateService.getRate(
      paymentData.fromCurrency,
      paymentData.toCurrency
    );
    
    // Convert amount
    const convertedAmount = paymentData.amount * exchangeRate.rate;
    
    // Check regulatory compliance
    await this.complianceService.checkInternationalTransfer(paymentData);
    
    // Process payment with converted amount
    const payment = await this.paymentService.processPayment({
      ...paymentData,
      amount: convertedAmount,
      currency: paymentData.toCurrency,
      exchangeRate: exchangeRate.rate,
      originalAmount: paymentData.amount,
      originalCurrency: paymentData.fromCurrency
    });
    
    // Log conversion for audit
    await this.database.currencyConversions.insert({
      paymentId: payment.id,
      fromCurrency: paymentData.fromCurrency,
      toCurrency: paymentData.toCurrency,
      originalAmount: paymentData.amount,
      convertedAmount: convertedAmount,
      exchangeRate: exchangeRate.rate,
      timestamp: new Date()
    });
    
    return payment;
  }
  
  async getExchangeRates(baseCurrency) {
    const rates = await this.exchangeRateService.getAllRates(baseCurrency);
    
    return {
      base: baseCurrency,
      rates: rates,
      timestamp: new Date()
    };
  }
}
```

### **Advanced Security & Fraud Prevention**

**Q4: How do you implement 3D Secure authentication?**
**A:**
```javascript
// 3D Secure Authentication Service
class ThreeDSecureService {
  async initiate3DSecure(paymentData) {
    // Check if 3DS is required
    const requires3DS = await this.check3DSRequirement(paymentData);
    
    if (!requires3DS) {
      return { requires3DS: false };
    }
    
    // Initiate 3DS authentication
    const authRequest = {
      amount: paymentData.amount,
      currency: paymentData.currency,
      cardNumber: paymentData.cardNumber,
      merchantId: paymentData.merchantId,
      returnUrl: paymentData.returnUrl
    };
    
    const authResponse = await this.paymentProvider.initiate3DS(authRequest);
    
    // Store authentication session
    await this.database.threeDSSessions.insert({
      sessionId: authResponse.sessionId,
      paymentId: paymentData.id,
      status: 'pending',
      acsUrl: authResponse.acsUrl,
      paReq: authResponse.paReq,
      createdAt: new Date()
    });
    
    return {
      requires3DS: true,
      acsUrl: authResponse.acsUrl,
      paReq: authResponse.paReq,
      sessionId: authResponse.sessionId
    };
  }
  
  async handle3DSecureCallback(callbackData) {
    const session = await this.database.threeDSSessions.findBySessionId(callbackData.sessionId);
    
    if (!session) {
      throw new Error('Invalid 3DS session');
    }
    
    // Verify authentication response
    const verificationResult = await this.paymentProvider.verify3DS({
      sessionId: callbackData.sessionId,
      paRes: callbackData.paRes
    });
    
    if (verificationResult.success) {
      // Complete payment
      const payment = await this.paymentService.completePayment(session.paymentId);
      session.status = 'completed';
    } else {
      session.status = 'failed';
      session.error = verificationResult.error;
    }
    
    await this.database.threeDSSessions.update(session.id, session);
    
    return verificationResult;
  }
}
```

**Q5: How do you implement real-time fraud detection?**
**A:**
```javascript
// Real-time Fraud Detection Service
class FraudDetectionService {
  async analyzeTransaction(transaction) {
    const riskFactors = [];
    let riskScore = 0;
    
    // Check transaction velocity
    const velocityCheck = await this.checkTransactionVelocity(transaction);
    if (velocityCheck.risk > 0.7) {
      riskFactors.push('High transaction velocity');
      riskScore += 30;
    }
    
    // Check device fingerprint
    const deviceCheck = await this.checkDeviceFingerprint(transaction);
    if (deviceCheck.risk > 0.8) {
      riskFactors.push('Suspicious device');
      riskScore += 25;
    }
    
    // Check geolocation
    const locationCheck = await this.checkGeolocation(transaction);
    if (locationCheck.risk > 0.6) {
      riskFactors.push('Unusual location');
      riskScore += 20;
    }
    
    // Check amount patterns
    const amountCheck = await this.checkAmountPatterns(transaction);
    if (amountCheck.risk > 0.5) {
      riskFactors.push('Unusual amount');
      riskScore += 15;
    }
    
    // Check blacklist
    const blacklistCheck = await this.checkBlacklist(transaction);
    if (blacklistCheck.isBlacklisted) {
      riskFactors.push('Blacklisted entity');
      riskScore += 50;
    }
    
    const riskLevel = this.calculateRiskLevel(riskScore);
    
    return {
      riskScore,
      riskLevel,
      riskFactors,
      recommendation: this.getRecommendation(riskLevel)
    };
  }
  
  async checkTransactionVelocity(transaction) {
    const recentTransactions = await this.database.transactions.findRecentByUser(
      transaction.customerId,
      24 // hours
    );
    
    const transactionCount = recentTransactions.length;
    const totalAmount = recentTransactions.reduce((sum, t) => sum + t.amount, 0);
    
    // Calculate velocity risk
    let risk = 0;
    if (transactionCount > 10) risk += 0.3;
    if (totalAmount > 10000) risk += 0.4;
    if (transaction.amount > 5000) risk += 0.3;
    
    return { risk: Math.min(risk, 1) };
  }
}
```

### **Compliance & Reporting**

**Q6: How do you implement AML (Anti-Money Laundering) compliance?**
**A:**
```javascript
// AML Compliance Service
class AMLComplianceService {
  async checkAMLCompliance(transaction) {
    const amlChecks = [];
    
    // Check transaction amount thresholds
    if (transaction.amount > 10000) {
      amlChecks.push(await this.performEnhancedDueDiligence(transaction));
    }
    
    // Check customer risk profile
    const customerRisk = await this.assessCustomerRisk(transaction.customerId);
    if (customerRisk.level === 'high') {
      amlChecks.push(await this.performCustomerScreening(transaction));
    }
    
    // Check for suspicious patterns
    const patternCheck = await this.checkSuspiciousPatterns(transaction);
    if (patternCheck.suspicious) {
      amlChecks.push(patternCheck);
    }
    
    // Check sanctions lists
    const sanctionsCheck = await this.checkSanctionsLists(transaction);
    if (sanctionsCheck.match) {
      amlChecks.push(sanctionsCheck);
    }
    
    const complianceResult = {
      transactionId: transaction.id,
      checks: amlChecks,
      overallRisk: this.calculateOverallRisk(amlChecks),
      requiresReporting: this.requiresSAR(amlChecks),
      timestamp: new Date()
    };
    
    // Store compliance result
    await this.database.amlChecks.insert(complianceResult);
    
    return complianceResult;
  }
  
  async generateSAR(suspiciousTransaction) {
    const sar = {
      id: this.generateID(),
      transactionId: suspiciousTransaction.id,
      customerId: suspiciousTransaction.customerId,
      amount: suspiciousTransaction.amount,
      reason: suspiciousTransaction.amlResult.reason,
      riskLevel: suspiciousTransaction.amlResult.overallRisk,
      generatedAt: new Date(),
      status: 'pending'
    };
    
    await this.database.sars.insert(sar);
    
    // Submit to regulatory authority
    await this.regulatoryService.submitSAR(sar);
    
    return sar;
  }
}
```

**Q7: How do you implement payment reconciliation and settlement?**
**A:**
```javascript
// Payment Reconciliation Service
class PaymentReconciliationService {
  async performDailyReconciliation(date) {
    const reconciliation = {
      id: this.generateID(),
      date: date,
      status: 'in_progress',
      startedAt: new Date()
    };
    
    await this.database.reconciliations.insert(reconciliation);
    
    try {
      // Get all transactions for the date
      const transactions = await this.database.transactions.findByDate(date);
      
      // Get provider settlement data
      const providerData = await this.paymentProvider.getSettlementData(date);
      
      // Perform reconciliation
      const reconciliationResult = await this.reconcileTransactions(transactions, providerData);
      
      // Update reconciliation status
      reconciliation.status = 'completed';
      reconciliation.completedAt = new Date();
      reconciliation.result = reconciliationResult;
      
      await this.database.reconciliations.update(reconciliation.id, reconciliation);
      
      // Handle discrepancies
      if (reconciliationResult.discrepancies.length > 0) {
        await this.handleDiscrepancies(reconciliationResult.discrepancies);
      }
      
      return reconciliation;
      
    } catch (error) {
      reconciliation.status = 'failed';
      reconciliation.error = error.message;
      await this.database.reconciliations.update(reconciliation.id, reconciliation);
      throw error;
    }
  }
  
  async reconcileTransactions(transactions, providerData) {
    const matched = [];
    const unmatched = [];
    const discrepancies = [];
    
    // Match transactions with provider data
    for (const transaction of transactions) {
      const providerTransaction = providerData.find(p => p.reference === transaction.id);
      
      if (providerTransaction) {
        if (transaction.amount === providerTransaction.amount) {
          matched.push({ transaction, providerTransaction });
        } else {
          discrepancies.push({
            transaction,
            providerTransaction,
            type: 'amount_mismatch',
            difference: transaction.amount - providerTransaction.amount
          });
        }
      } else {
        unmatched.push({ transaction, type: 'missing_in_provider' });
      }
    }
    
    return {
      matched: matched.length,
      unmatched: unmatched.length,
      discrepancies: discrepancies.length,
      details: { matched, unmatched, discrepancies }
    };
  }
}
```
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
