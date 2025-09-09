# 🔒 Node.js Security: Complete Guide

> **Master Node.js security best practices for production applications**

## 🎯 **Learning Objectives**

- Master Node.js security fundamentals
- Implement authentication and authorization
- Learn secure coding practices
- Understand common vulnerabilities and attacks
- Build secure production applications

## 📚 **Table of Contents**

1. [Security Fundamentals](#security-fundamentals)
2. [Authentication & Authorization](#authentication--authorization)
3. [Input Validation & Sanitization](#input-validation--sanitization)
4. [Data Protection](#data-protection)
5. [Network Security](#network-security)
6. [Dependency Security](#dependency-security)
7. [Security Headers](#security-headers)
8. [Monitoring & Logging](#monitoring--logging)
9. [Interview Questions](#interview-questions)

---

## 🚀 **Security Fundamentals**

### **What is Application Security?**

Application security refers to the measures taken to protect applications from threats and vulnerabilities. In Node.js, this includes:

- **Authentication**: Verifying user identity
- **Authorization**: Controlling access to resources
- **Data Protection**: Encrypting sensitive data
- **Input Validation**: Sanitizing user inputs
- **Network Security**: Securing communications
- **Dependency Management**: Keeping packages secure

### **Security Principles**

```javascript
// Security Principles Implementation
class SecurityPrinciples {
    constructor() {
        this.principles = {
            defenseInDepth: 'Multiple layers of security',
            leastPrivilege: 'Minimum necessary permissions',
            failSecure: 'Fail in a secure state',
            separationOfDuties: 'Different people for different tasks',
            securityByDesign: 'Security from the beginning',
            regularUpdates: 'Keep dependencies updated'
        };
    }
    
    // Defense in Depth
    implementDefenseInDepth() {
        return {
            network: 'Firewalls, VPNs, DDoS protection',
            application: 'Input validation, authentication, authorization',
            data: 'Encryption at rest and in transit',
            monitoring: 'Logging, alerting, intrusion detection'
        };
    }
    
    // Least Privilege
    implementLeastPrivilege(user, resource) {
        const permissions = this.getUserPermissions(user);
        const requiredPermissions = this.getResourcePermissions(resource);
        
        return this.hasMinimumPermissions(permissions, requiredPermissions);
    }
    
    // Fail Secure
    failSecure(error, context) {
        // Log the error
        console.error('Security error:', error, context);
        
        // Return safe default
        return {
            success: false,
            message: 'Access denied',
            data: null
        };
    }
}
```

### **Threat Modeling**

```javascript
// Threat Modeling for Node.js Applications
class ThreatModeling {
    constructor() {
        this.threats = {
            injection: 'SQL injection, NoSQL injection, command injection',
            brokenAuth: 'Weak authentication, session management',
            sensitiveData: 'Insecure data storage, transmission',
            xmlExternal: 'XXE attacks, XML bomb',
            brokenAccess: 'Insecure direct object references',
            securityMisconfig: 'Default configurations, missing patches',
            xss: 'Cross-site scripting attacks',
            insecureDeserialization: 'Object injection, code execution',
            knownVulns: 'Outdated dependencies, known vulnerabilities',
            logging: 'Insufficient logging, monitoring'
        };
    }
    
    assessThreats(application) {
        const threats = [];
        
        // Check for injection vulnerabilities
        if (this.hasDatabaseQueries(application)) {
            threats.push({
                type: 'injection',
                severity: 'high',
                mitigation: 'Use parameterized queries, input validation'
            });
        }
        
        // Check for authentication issues
        if (this.hasWeakAuth(application)) {
            threats.push({
                type: 'brokenAuth',
                severity: 'high',
                mitigation: 'Strong passwords, multi-factor authentication'
            });
        }
        
        // Check for data protection
        if (this.hasUnencryptedData(application)) {
            threats.push({
                type: 'sensitiveData',
                severity: 'high',
                mitigation: 'Encrypt data at rest and in transit'
            });
        }
        
        return threats;
    }
    
    hasDatabaseQueries(app) {
        // Check if app uses raw database queries
        return app.includes('SELECT') || app.includes('INSERT');
    }
    
    hasWeakAuth(app) {
        // Check for weak authentication patterns
        return app.includes('password') && !app.includes('bcrypt');
    }
    
    hasUnencryptedData(app) {
        // Check for unencrypted sensitive data
        return app.includes('creditCard') && !app.includes('encrypt');
    }
}
```

---

## 🔐 **Authentication & Authorization**

### **JWT Authentication**

```javascript
// JWT Authentication Implementation
class JWTAuthentication {
    constructor(secretKey) {
        this.jwt = require('jsonwebtoken');
        this.secretKey = secretKey;
        this.refreshTokens = new Map();
        this.blacklistedTokens = new Set();
    }
    
    // Generate access token
    generateAccessToken(payload) {
        return this.jwt.sign(payload, this.secretKey, {
            expiresIn: '15m',
            issuer: 'your-app',
            audience: 'your-app-users'
        });
    }
    
    // Generate refresh token
    generateRefreshToken(payload) {
        const refreshToken = this.jwt.sign(payload, this.secretKey, {
            expiresIn: '7d',
            issuer: 'your-app',
            audience: 'your-app-users'
        });
        
        // Store refresh token
        this.refreshTokens.set(refreshToken, {
            userId: payload.userId,
            createdAt: Date.now()
        });
        
        return refreshToken;
    }
    
    // Verify token
    verifyToken(token) {
        try {
            const decoded = this.jwt.verify(token, this.secretKey, {
                issuer: 'your-app',
                audience: 'your-app-users'
            });
            
            // Check if token is blacklisted
            if (this.blacklistedTokens.has(token)) {
                throw new Error('Token is blacklisted');
            }
            
            return decoded;
        } catch (error) {
            throw new Error('Invalid token');
        }
    }
    
    // Refresh access token
    refreshAccessToken(refreshToken) {
        if (!this.refreshTokens.has(refreshToken)) {
            throw new Error('Invalid refresh token');
        }
        
        const tokenData = this.refreshTokens.get(refreshToken);
        const newAccessToken = this.generateAccessToken({
            userId: tokenData.userId
        });
        
        return newAccessToken;
    }
    
    // Revoke token
    revokeToken(token) {
        this.blacklistedTokens.add(token);
        this.refreshTokens.delete(token);
    }
    
    // Middleware for protecting routes
    authenticateToken(req, res, next) {
        const authHeader = req.headers['authorization'];
        const token = authHeader && authHeader.split(' ')[1];
        
        if (!token) {
            return res.status(401).json({ error: 'Access token required' });
        }
        
        try {
            const decoded = this.verifyToken(token);
            req.user = decoded;
            next();
        } catch (error) {
            return res.status(403).json({ error: 'Invalid token' });
        }
    }
}
```

### **Password Security**

```javascript
// Password Security Implementation
class PasswordSecurity {
    constructor() {
        this.bcrypt = require('bcrypt');
        this.saltRounds = 12;
        this.passwordPolicy = {
            minLength: 8,
            requireUppercase: true,
            requireLowercase: true,
            requireNumbers: true,
            requireSpecialChars: true,
            maxLength: 128
        };
    }
    
    // Hash password
    async hashPassword(password) {
        const salt = await this.bcrypt.genSalt(this.saltRounds);
        return await this.bcrypt.hash(password, salt);
    }
    
    // Verify password
    async verifyPassword(password, hash) {
        return await this.bcrypt.compare(password, hash);
    }
    
    // Validate password strength
    validatePassword(password) {
        const errors = [];
        
        if (password.length < this.passwordPolicy.minLength) {
            errors.push(`Password must be at least ${this.passwordPolicy.minLength} characters long`);
        }
        
        if (password.length > this.passwordPolicy.maxLength) {
            errors.push(`Password must be no more than ${this.passwordPolicy.maxLength} characters long`);
        }
        
        if (this.passwordPolicy.requireUppercase && !/[A-Z]/.test(password)) {
            errors.push('Password must contain at least one uppercase letter');
        }
        
        if (this.passwordPolicy.requireLowercase && !/[a-z]/.test(password)) {
            errors.push('Password must contain at least one lowercase letter');
        }
        
        if (this.passwordPolicy.requireNumbers && !/\d/.test(password)) {
            errors.push('Password must contain at least one number');
        }
        
        if (this.passwordPolicy.requireSpecialChars && !/[!@#$%^&*(),.?":{}|<>]/.test(password)) {
            errors.push('Password must contain at least one special character');
        }
        
        return {
            isValid: errors.length === 0,
            errors: errors
        };
    }
    
    // Check if password is in common passwords list
    isCommonPassword(password) {
        const commonPasswords = [
            'password', '123456', '123456789', 'qwerty', 'abc123',
            'password123', 'admin', 'letmein', 'welcome', 'monkey'
        ];
        
        return commonPasswords.includes(password.toLowerCase());
    }
    
    // Generate secure password
    generateSecurePassword(length = 16) {
        const charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*';
        let password = '';
        
        for (let i = 0; i < length; i++) {
            password += charset.charAt(Math.floor(Math.random() * charset.length));
        }
        
        return password;
    }
}
```

### **Role-Based Access Control**

```javascript
// Role-Based Access Control (RBAC)
class RBAC {
    constructor() {
        this.roles = new Map();
        this.permissions = new Map();
        this.userRoles = new Map();
    }
    
    // Define roles
    defineRole(roleName, permissions) {
        this.roles.set(roleName, {
            name: roleName,
            permissions: permissions,
            createdAt: Date.now()
        });
    }
    
    // Define permissions
    definePermission(permissionName, resource, action) {
        this.permissions.set(permissionName, {
            name: permissionName,
            resource: resource,
            action: action
        });
    }
    
    // Assign role to user
    assignRole(userId, roleName) {
        if (!this.roles.has(roleName)) {
            throw new Error(`Role ${roleName} does not exist`);
        }
        
        if (!this.userRoles.has(userId)) {
            this.userRoles.set(userId, []);
        }
        
        const userRoles = this.userRoles.get(userId);
        if (!userRoles.includes(roleName)) {
            userRoles.push(roleName);
        }
    }
    
    // Check if user has permission
    hasPermission(userId, permissionName) {
        const userRoles = this.userRoles.get(userId) || [];
        
        for (const roleName of userRoles) {
            const role = this.roles.get(roleName);
            if (role && role.permissions.includes(permissionName)) {
                return true;
            }
        }
        
        return false;
    }
    
    // Check if user has role
    hasRole(userId, roleName) {
        const userRoles = this.userRoles.get(userId) || [];
        return userRoles.includes(roleName);
    }
    
    // Get user permissions
    getUserPermissions(userId) {
        const userRoles = this.userRoles.get(userId) || [];
        const permissions = new Set();
        
        for (const roleName of userRoles) {
            const role = this.roles.get(roleName);
            if (role) {
                role.permissions.forEach(permission => permissions.add(permission));
            }
        }
        
        return Array.from(permissions);
    }
    
    // Middleware for role-based access
    requireRole(roleName) {
        return (req, res, next) => {
            const userId = req.user?.userId;
            
            if (!userId) {
                return res.status(401).json({ error: 'Authentication required' });
            }
            
            if (!this.hasRole(userId, roleName)) {
                return res.status(403).json({ error: 'Insufficient permissions' });
            }
            
            next();
        };
    }
    
    // Middleware for permission-based access
    requirePermission(permissionName) {
        return (req, res, next) => {
            const userId = req.user?.userId;
            
            if (!userId) {
                return res.status(401).json({ error: 'Authentication required' });
            }
            
            if (!this.hasPermission(userId, permissionName)) {
                return res.status(403).json({ error: 'Insufficient permissions' });
            }
            
            next();
        };
    }
}
```

---

## 🛡️ **Input Validation & Sanitization**

### **Input Validation**

```javascript
// Input Validation System
class InputValidator {
    constructor() {
        this.schemas = new Map();
        this.sanitizers = new Map();
    }
    
    // Define validation schema
    defineSchema(name, schema) {
        this.schemas.set(name, schema);
    }
    
    // Validate data against schema
    validate(data, schemaName) {
        const schema = this.schemas.get(schemaName);
        if (!schema) {
            throw new Error(`Schema ${schemaName} not found`);
        }
        
        const errors = [];
        
        for (const [field, rules] of Object.entries(schema)) {
            const value = data[field];
            
            // Required validation
            if (rules.required && (value === undefined || value === null || value === '')) {
                errors.push(`${field} is required`);
                continue;
            }
            
            // Skip validation if value is not provided and not required
            if (value === undefined || value === null) {
                continue;
            }
            
            // Type validation
            if (rules.type && typeof value !== rules.type) {
                errors.push(`${field} must be of type ${rules.type}`);
            }
            
            // String validations
            if (rules.type === 'string') {
                if (rules.minLength && value.length < rules.minLength) {
                    errors.push(`${field} must be at least ${rules.minLength} characters long`);
                }
                
                if (rules.maxLength && value.length > rules.maxLength) {
                    errors.push(`${field} must be no more than ${rules.maxLength} characters long`);
                }
                
                if (rules.pattern && !rules.pattern.test(value)) {
                    errors.push(`${field} format is invalid`);
                }
                
                if (rules.enum && !rules.enum.includes(value)) {
                    errors.push(`${field} must be one of: ${rules.enum.join(', ')}`);
                }
            }
            
            // Number validations
            if (rules.type === 'number') {
                if (rules.min !== undefined && value < rules.min) {
                    errors.push(`${field} must be at least ${rules.min}`);
                }
                
                if (rules.max !== undefined && value > rules.max) {
                    errors.push(`${field} must be no more than ${rules.max}`);
                }
            }
            
            // Array validations
            if (rules.type === 'array') {
                if (!Array.isArray(value)) {
                    errors.push(`${field} must be an array`);
                } else {
                    if (rules.minItems && value.length < rules.minItems) {
                        errors.push(`${field} must have at least ${rules.minItems} items`);
                    }
                    
                    if (rules.maxItems && value.length > rules.maxItems) {
                        errors.push(`${field} must have no more than ${rules.maxItems} items`);
                    }
                }
            }
        }
        
        return {
            isValid: errors.length === 0,
            errors: errors
        };
    }
    
    // Sanitize data
    sanitize(data, sanitizerName) {
        const sanitizer = this.sanitizers.get(sanitizerName);
        if (!sanitizer) {
            throw new Error(`Sanitizer ${sanitizerName} not found`);
        }
        
        return sanitizer(data);
    }
    
    // Define sanitizer
    defineSanitizer(name, sanitizer) {
        this.sanitizers.set(name, sanitizer);
    }
}
```

### **XSS Protection**

```javascript
// XSS Protection
class XSSProtection {
    constructor() {
        this.xss = require('xss');
        this.dompurify = require('dompurify');
        this.jsdom = require('jsdom');
    }
    
    // Sanitize HTML
    sanitizeHTML(html) {
        const { JSDOM } = this.jsdom;
        const window = new JSDOM('').window;
        const purify = this.dompurify(window);
        
        return purify.sanitize(html);
    }
    
    // Sanitize text
    sanitizeText(text) {
        return this.xss(text, {
            whiteList: {}, // No HTML tags allowed
            stripIgnoreTag: true,
            stripIgnoreTagBody: ['script']
        });
    }
    
    // Sanitize user input
    sanitizeInput(input) {
        if (typeof input === 'string') {
            return this.sanitizeText(input);
        } else if (typeof input === 'object' && input !== null) {
            const sanitized = {};
            for (const [key, value] of Object.entries(input)) {
                sanitized[key] = this.sanitizeInput(value);
            }
            return sanitized;
        }
        return input;
    }
    
    // Escape HTML entities
    escapeHTML(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        };
        
        return text.replace(/[&<>"']/g, (m) => map[m]);
    }
    
    // Validate URL
    validateURL(url) {
        try {
            const parsed = new URL(url);
            const allowedProtocols = ['http:', 'https:'];
            
            if (!allowedProtocols.includes(parsed.protocol)) {
                return false;
            }
            
            return true;
        } catch (error) {
            return false;
        }
    }
}
```

### **SQL Injection Protection**

```javascript
// SQL Injection Protection
class SQLInjectionProtection {
    constructor() {
        this.pg = require('pg');
        this.mysql = require('mysql2');
    }
    
    // Use parameterized queries (PostgreSQL)
    async safeQueryPostgreSQL(query, params) {
        const client = new this.pg.Client();
        await client.connect();
        
        try {
            const result = await client.query(query, params);
            return result.rows;
        } finally {
            await client.end();
        }
    }
    
    // Use parameterized queries (MySQL)
    async safeQueryMySQL(query, params) {
        const connection = this.mysql.createConnection({
            host: process.env.DB_HOST,
            user: process.env.DB_USER,
            password: process.env.DB_PASSWORD,
            database: process.env.DB_NAME
        });
        
        return new Promise((resolve, reject) => {
            connection.execute(query, params, (error, results) => {
                connection.end();
                if (error) {
                    reject(error);
                } else {
                    resolve(results);
                }
            });
        });
    }
    
    // Validate input for SQL injection
    validateInput(input) {
        const dangerousPatterns = [
            /(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)/i,
            /(\b(OR|AND)\s+\d+\s*=\s*\d+)/i,
            /(\b(OR|AND)\s+['"]\s*=\s*['"])/i,
            /(\b(OR|AND)\s+['"]\s*LIKE\s*['"])/i,
            /(\b(OR|AND)\s+['"]\s*IN\s*\(/i,
            /(\b(OR|AND)\s+['"]\s*BETWEEN\s+)/i,
            /(\b(OR|AND)\s+['"]\s*IS\s+NULL)/i,
            /(\b(OR|AND)\s+['"]\s*IS\s+NOT\s+NULL)/i,
            /(\b(OR|AND)\s+['"]\s*EXISTS\s*\(/i,
            /(\b(OR|AND)\s+['"]\s*NOT\s+EXISTS\s*\(/i
        ];
        
        for (const pattern of dangerousPatterns) {
            if (pattern.test(input)) {
                throw new Error('Potential SQL injection detected');
            }
        }
        
        return true;
    }
    
    // Sanitize input for SQL
    sanitizeSQLInput(input) {
        if (typeof input !== 'string') {
            return input;
        }
        
        // Remove or escape dangerous characters
        return input
            .replace(/['"\\]/g, '\\$&')
            .replace(/\0/g, '\\0')
            .replace(/\n/g, '\\n')
            .replace(/\r/g, '\\r')
            .replace(/\x1a/g, '\\Z');
    }
}
```

---

## 🔒 **Data Protection**

### **Encryption**

```javascript
// Data Encryption
class DataEncryption {
    constructor() {
        this.crypto = require('crypto');
        this.algorithm = 'aes-256-gcm';
        this.keyLength = 32;
        this.ivLength = 16;
        this.tagLength = 16;
    }
    
    // Generate encryption key
    generateKey() {
        return this.crypto.randomBytes(this.keyLength);
    }
    
    // Encrypt data
    encrypt(data, key) {
        const iv = this.crypto.randomBytes(this.ivLength);
        const cipher = this.crypto.createCipher(this.algorithm, key);
        cipher.setAAD(Buffer.from('additional data'));
        
        let encrypted = cipher.update(data, 'utf8', 'hex');
        encrypted += cipher.final('hex');
        
        const tag = cipher.getAuthTag();
        
        return {
            encrypted: encrypted,
            iv: iv.toString('hex'),
            tag: tag.toString('hex')
        };
    }
    
    // Decrypt data
    decrypt(encryptedData, key) {
        const decipher = this.crypto.createDecipher(this.algorithm, key);
        decipher.setAAD(Buffer.from('additional data'));
        decipher.setAuthTag(Buffer.from(encryptedData.tag, 'hex'));
        
        let decrypted = decipher.update(encryptedData.encrypted, 'hex', 'utf8');
        decrypted += decipher.final('utf8');
        
        return decrypted;
    }
    
    // Hash data
    hash(data, algorithm = 'sha256') {
        return this.crypto.createHash(algorithm).update(data).digest('hex');
    }
    
    // Generate HMAC
    generateHMAC(data, key) {
        return this.crypto.createHmac('sha256', key).update(data).digest('hex');
    }
    
    // Verify HMAC
    verifyHMAC(data, key, hmac) {
        const expectedHmac = this.generateHMAC(data, key);
        return this.crypto.timingSafeEqual(Buffer.from(hmac), Buffer.from(expectedHmac));
    }
    
    // Generate secure random string
    generateSecureRandom(length = 32) {
        return this.crypto.randomBytes(length).toString('hex');
    }
}
```

### **Secrets Management**

```javascript
// Secrets Management
class SecretsManager {
    constructor() {
        this.secrets = new Map();
        this.encryption = new DataEncryption();
        this.masterKey = process.env.MASTER_KEY || this.encryption.generateKey();
    }
    
    // Store secret
    storeSecret(key, value) {
        const encrypted = this.encryption.encrypt(value, this.masterKey);
        this.secrets.set(key, encrypted);
    }
    
    // Retrieve secret
    retrieveSecret(key) {
        const encrypted = this.secrets.get(key);
        if (!encrypted) {
            throw new Error(`Secret ${key} not found`);
        }
        
        return this.encryption.decrypt(encrypted, this.masterKey);
    }
    
    // Delete secret
    deleteSecret(key) {
        this.secrets.delete(key);
    }
    
    // List secrets
    listSecrets() {
        return Array.from(this.secrets.keys());
    }
    
    // Rotate master key
    rotateMasterKey() {
        const newMasterKey = this.encryption.generateKey();
        const newSecrets = new Map();
        
        // Re-encrypt all secrets with new key
        for (const [key, encrypted] of this.secrets) {
            const decrypted = this.encryption.decrypt(encrypted, this.masterKey);
            const reEncrypted = this.encryption.encrypt(decrypted, newMasterKey);
            newSecrets.set(key, reEncrypted);
        }
        
        this.masterKey = newMasterKey;
        this.secrets = newSecrets;
    }
}
```

---

## 🌐 **Network Security**

### **HTTPS Configuration**

```javascript
// HTTPS Configuration
class HTTPSConfiguration {
    constructor() {
        this.https = require('https');
        this.fs = require('fs');
    }
    
    // Create HTTPS server
    createHTTPSServer(app, options) {
        const serverOptions = {
            key: this.fs.readFileSync(options.keyPath),
            cert: this.fs.readFileSync(options.certPath),
            ...options
        };
        
        return this.https.createServer(serverOptions, app);
    }
    
    // Redirect HTTP to HTTPS
    redirectToHTTPS(req, res, next) {
        if (req.secure) {
            next();
        } else {
            res.redirect(301, `https://${req.headers.host}${req.url}`);
        }
    }
    
    // HSTS headers
    setHSTSHeaders(req, res, next) {
        res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains; preload');
        next();
    }
}
```

### **Rate Limiting**

```javascript
// Rate Limiting
class RateLimiter {
    constructor() {
        this.requests = new Map();
        this.windows = new Map();
    }
    
    // Check if request is allowed
    isAllowed(identifier, limit = 100, windowMs = 60000) {
        const now = Date.now();
        const windowStart = now - windowMs;
        
        // Clean old requests
        if (this.requests.has(identifier)) {
            const userRequests = this.requests.get(identifier);
            const validRequests = userRequests.filter(time => time > windowStart);
            this.requests.set(identifier, validRequests);
        } else {
            this.requests.set(identifier, []);
        }
        
        const userRequests = this.requests.get(identifier);
        
        if (userRequests.length >= limit) {
            return false;
        }
        
        userRequests.push(now);
        return true;
    }
    
    // Middleware for rate limiting
    rateLimit(limit = 100, windowMs = 60000) {
        return (req, res, next) => {
            const identifier = req.ip || req.connection.remoteAddress;
            
            if (!this.isAllowed(identifier, limit, windowMs)) {
                return res.status(429).json({
                    error: 'Too many requests',
                    retryAfter: Math.ceil(windowMs / 1000)
                });
            }
            
            next();
        };
    }
    
    // Different limits for different endpoints
    createRateLimit(limits) {
        return (req, res, next) => {
            const path = req.path;
            const limit = limits[path] || limits.default || 100;
            
            const identifier = req.ip || req.connection.remoteAddress;
            
            if (!this.isAllowed(identifier, limit.limit, limit.windowMs)) {
                return res.status(429).json({
                    error: 'Too many requests',
                    retryAfter: Math.ceil(limit.windowMs / 1000)
                });
            }
            
            next();
        };
    }
}
```

---

## 🔍 **Dependency Security**

### **Vulnerability Scanning**

```javascript
// Dependency Security Scanner
class DependencyScanner {
    constructor() {
        this.npm = require('npm');
        this.audit = require('npm-audit');
    }
    
    // Scan for vulnerabilities
    async scanVulnerabilities() {
        try {
            const auditResult = await this.audit();
            return this.parseAuditResult(auditResult);
        } catch (error) {
            console.error('Audit failed:', error);
            return null;
        }
    }
    
    // Parse audit results
    parseAuditResult(auditResult) {
        const vulnerabilities = [];
        
        if (auditResult.vulnerabilities) {
            for (const [packageName, vuln] of Object.entries(auditResult.vulnerabilities)) {
                vulnerabilities.push({
                    package: packageName,
                    severity: vuln.severity,
                    title: vuln.title,
                    description: vuln.description,
                    recommendation: vuln.recommendation
                });
            }
        }
        
        return {
            vulnerabilities: vulnerabilities,
            summary: auditResult.metadata?.vulnerabilities || {}
        };
    }
    
    // Check for outdated packages
    async checkOutdatedPackages() {
        try {
            const outdated = await this.npm.outdated();
            return outdated;
        } catch (error) {
            console.error('Outdated check failed:', error);
            return null;
        }
    }
    
    // Update packages
    async updatePackages(packages) {
        try {
            const updateResult = await this.npm.update(packages);
            return updateResult;
        } catch (error) {
            console.error('Update failed:', error);
            return null;
        }
    }
}
```

---

## 🎯 **Interview Questions**

### **1. What are the main security vulnerabilities in Node.js applications?**

**Answer:**
- **Injection Attacks**: SQL injection, NoSQL injection, command injection
- **Broken Authentication**: Weak passwords, session management issues
- **Sensitive Data Exposure**: Unencrypted data, insecure transmission
- **XML External Entities**: XXE attacks, XML bombs
- **Broken Access Control**: Insecure direct object references
- **Security Misconfiguration**: Default configurations, missing patches
- **Cross-Site Scripting**: XSS attacks
- **Insecure Deserialization**: Object injection, code execution
- **Known Vulnerabilities**: Outdated dependencies
- **Insufficient Logging**: Poor monitoring and logging

### **2. How do you implement secure authentication in Node.js?**

**Answer:**
- **Password Security**: Use bcrypt for hashing, strong password policies
- **JWT Tokens**: Secure token generation and validation
- **Session Management**: Secure session storage and handling
- **Multi-Factor Authentication**: Additional security layers
- **Rate Limiting**: Prevent brute force attacks
- **Account Lockout**: Lock accounts after failed attempts
- **Password Reset**: Secure password reset mechanisms

### **3. How do you prevent SQL injection in Node.js?**

**Answer:**
- **Parameterized Queries**: Use prepared statements
- **Input Validation**: Validate and sanitize all inputs
- **ORM/Query Builder**: Use libraries that handle SQL safely
- **Least Privilege**: Database users with minimal permissions
- **Regular Updates**: Keep database drivers updated
- **Code Review**: Regular security code reviews

### **4. What are security headers and why are they important?**

**Answer:**
Security headers help protect against various attacks:
- **Content-Security-Policy**: Prevents XSS attacks
- **X-Frame-Options**: Prevents clickjacking
- **X-Content-Type-Options**: Prevents MIME type sniffing
- **Strict-Transport-Security**: Enforces HTTPS
- **X-XSS-Protection**: Enables browser XSS protection
- **Referrer-Policy**: Controls referrer information

### **5. How do you handle sensitive data in Node.js?**

**Answer:**
- **Encryption**: Encrypt data at rest and in transit
- **Secrets Management**: Use secure secret storage
- **Environment Variables**: Store sensitive config in env vars
- **Key Management**: Secure key generation and rotation
- **Data Masking**: Mask sensitive data in logs
- **Access Control**: Limit access to sensitive data
- **Audit Logging**: Log access to sensitive data

---

**🎉 Node.js security is crucial for protecting applications and user data!**
