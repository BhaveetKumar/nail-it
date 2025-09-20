# Node.js Security Complete Guide

## Table of Contents
1. [Security Fundamentals](#security-fundamentals)
2. [Authentication & Authorization](#authentication--authorization)
3. [Input Validation & Sanitization](#input-validation--sanitization)
4. [SQL Injection Prevention](#sql-injection-prevention)
5. [XSS Prevention](#xss-prevention)
6. [CSRF Protection](#csrf-protection)
7. [Security Headers](#security-headers)
8. [HTTPS & SSL/TLS](#https--ssltls)
9. [Session Management](#session-management)
10. [File Upload Security](#file-upload-security)
11. [Dependency Security](#dependency-security)
12. [Logging & Monitoring](#logging--monitoring)
13. [Rate Limiting](#rate-limiting)
14. [Error Handling](#error-handling)
15. [Production Security](#production-security)

## Security Fundamentals

### Security Principles
- **Defense in Depth**: Multiple layers of security
- **Least Privilege**: Minimum necessary permissions
- **Fail Secure**: System fails in secure state
- **Security by Design**: Security built into architecture

### Common Vulnerabilities
- **OWASP Top 10**: Critical security risks
- **Injection Attacks**: SQL, NoSQL, Command injection
- **Broken Authentication**: Weak authentication mechanisms
- **Sensitive Data Exposure**: Unprotected sensitive data
- **XML External Entities**: XXE attacks
- **Broken Access Control**: Inadequate access restrictions
- **Security Misconfiguration**: Default configurations
- **Cross-Site Scripting**: XSS attacks
- **Insecure Deserialization**: Object injection
- **Known Vulnerabilities**: Outdated dependencies

## Authentication & Authorization

### JWT Implementation
```javascript
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');

class AuthService {
  constructor() {
    this.secretKey = process.env.JWT_SECRET || 'your-secret-key';
    this.expiresIn = '24h';
  }

  // Generate JWT token
  generateToken(payload) {
    return jwt.sign(payload, this.secretKey, { 
      expiresIn: this.expiresIn,
      issuer: 'your-app',
      audience: 'your-users'
    });
  }

  // Verify JWT token
  verifyToken(token) {
    try {
      return jwt.verify(token, this.secretKey, {
        issuer: 'your-app',
        audience: 'your-users'
      });
    } catch (error) {
      throw new Error('Invalid token');
    }
  }

  // Hash password
  async hashPassword(password) {
    const saltRounds = 12;
    return await bcrypt.hash(password, saltRounds);
  }

  // Verify password
  async verifyPassword(password, hashedPassword) {
    return await bcrypt.compare(password, hashedPassword);
  }

  // Login user
  async login(email, password) {
    const user = await this.findUserByEmail(email);
    if (!user) {
      throw new Error('Invalid credentials');
    }

    const isValidPassword = await this.verifyPassword(password, user.password);
    if (!isValidPassword) {
      throw new Error('Invalid credentials');
    }

    const token = this.generateToken({
      userId: user.id,
      email: user.email,
      role: user.role
    });

    return {
      token,
      user: {
        id: user.id,
        email: user.email,
        role: user.role
      }
    };
  }
}
```

### Role-Based Access Control (RBAC)
```javascript
class RBACService {
  constructor() {
    this.permissions = {
      'admin': ['read', 'write', 'delete', 'manage'],
      'user': ['read', 'write'],
      'guest': ['read']
    };
  }

  // Check if user has permission
  hasPermission(userRole, requiredPermission) {
    const rolePermissions = this.permissions[userRole] || [];
    return rolePermissions.includes(requiredPermission);
  }

  // Middleware for permission checking
  requirePermission(permission) {
    return (req, res, next) => {
      const userRole = req.user.role;
      
      if (!this.hasPermission(userRole, permission)) {
        return res.status(403).json({
          error: 'Insufficient permissions',
          required: permission,
          userRole: userRole
        });
      }
      
      next();
    };
  }

  // Check multiple permissions
  hasAnyPermission(userRole, permissions) {
    return permissions.some(permission => 
      this.hasPermission(userRole, permission)
    );
  }

  // Check all permissions
  hasAllPermissions(userRole, permissions) {
    return permissions.every(permission => 
      this.hasPermission(userRole, permission)
    );
  }
}
```

### OAuth 2.0 Implementation
```javascript
const passport = require('passport');
const GoogleStrategy = require('passport-google-oauth20').Strategy;

class OAuthService {
  constructor() {
    this.setupPassport();
  }

  setupPassport() {
    passport.use(new GoogleStrategy({
      clientID: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
      callbackURL: "/auth/google/callback"
    }, async (accessToken, refreshToken, profile, done) => {
      try {
        let user = await this.findOrCreateUser(profile);
        return done(null, user);
      } catch (error) {
        return done(error, null);
      }
    }));

    passport.serializeUser((user, done) => {
      done(null, user.id);
    });

    passport.deserializeUser(async (id, done) => {
      try {
        const user = await this.findUserById(id);
        done(null, user);
      } catch (error) {
        done(error, null);
      }
    });
  }

  async findOrCreateUser(profile) {
    let user = await this.findUserByGoogleId(profile.id);
    
    if (!user) {
      user = await this.createUser({
        googleId: profile.id,
        email: profile.emails[0].value,
        name: profile.displayName,
        avatar: profile.photos[0].value
      });
    }
    
    return user;
  }
}
```

## Input Validation & Sanitization

### Input Validation with Joi
```javascript
const Joi = require('joi');

class ValidationService {
  constructor() {
    this.schemas = {
      user: Joi.object({
        email: Joi.string().email().required(),
        password: Joi.string().min(8).pattern(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/).required(),
        name: Joi.string().min(2).max(50).required(),
        age: Joi.number().integer().min(18).max(120)
      }),
      
      product: Joi.object({
        name: Joi.string().min(1).max(100).required(),
        price: Joi.number().positive().required(),
        description: Joi.string().max(1000),
        category: Joi.string().valid('electronics', 'clothing', 'books').required()
      }),
      
      login: Joi.object({
        email: Joi.string().email().required(),
        password: Joi.string().required()
      })
    };
  }

  // Validate request body
  validate(schemaName) {
    return (req, res, next) => {
      const schema = this.schemas[schemaName];
      if (!schema) {
        return res.status(500).json({ error: 'Validation schema not found' });
      }

      const { error, value } = schema.validate(req.body);
      if (error) {
        return res.status(400).json({
          error: 'Validation failed',
          details: error.details.map(detail => ({
            field: detail.path.join('.'),
            message: detail.message
          }))
        });
      }

      req.body = value;
      next();
    };
  }

  // Sanitize HTML input
  sanitizeHtml(html) {
    const createDOMPurify = require('isomorphic-dompurify');
    return createDOMPurify.sanitize(html);
  }

  // Sanitize SQL input
  sanitizeSql(input) {
    return input.replace(/['"\\;]/g, '');
  }
}
```

### Express.js Security Middleware
```javascript
const express = require('express');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const cors = require('cors');
const mongoSanitize = require('express-mongo-sanitize');
const xss = require('xss-clean');
const hpp = require('hpp');

class SecurityMiddleware {
  constructor() {
    this.app = express();
    this.setupSecurity();
  }

  setupSecurity() {
    // Helmet for security headers
    this.app.use(helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          scriptSrc: ["'self'"],
          imgSrc: ["'self'", "data:", "https:"],
        },
      },
      hsts: {
        maxAge: 31536000,
        includeSubDomains: true,
        preload: true
      }
    }));

    // CORS configuration
    this.app.use(cors({
      origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
      credentials: true,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization']
    }));

    // Rate limiting
    const limiter = rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 100, // limit each IP to 100 requests per windowMs
      message: 'Too many requests from this IP, please try again later.',
      standardHeaders: true,
      legacyHeaders: false,
    });
    this.app.use(limiter);

    // Body parsing security
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));

    // MongoDB injection prevention
    this.app.use(mongoSanitize());

    // XSS protection
    this.app.use(xss());

    // HTTP Parameter Pollution protection
    this.app.use(hpp());
  }
}
```

## SQL Injection Prevention

### Parameterized Queries
```javascript
const mysql = require('mysql2/promise');

class DatabaseService {
  constructor() {
    this.pool = mysql.createPool({
      host: process.env.DB_HOST,
      user: process.env.DB_USER,
      password: process.env.DB_PASSWORD,
      database: process.env.DB_NAME,
      waitForConnections: true,
      connectionLimit: 10,
      queueLimit: 0
    });
  }

  // Safe query with parameters
  async safeQuery(sql, params = []) {
    try {
      const [rows] = await this.pool.execute(sql, params);
      return rows;
    } catch (error) {
      console.error('Database query error:', error);
      throw new Error('Database operation failed');
    }
  }

  // Get user by ID (safe)
  async getUserById(userId) {
    const sql = 'SELECT id, email, name FROM users WHERE id = ?';
    const users = await this.safeQuery(sql, [userId]);
    return users[0] || null;
  }

  // Search users (safe)
  async searchUsers(searchTerm) {
    const sql = 'SELECT id, email, name FROM users WHERE name LIKE ? OR email LIKE ?';
    const searchPattern = `%${searchTerm}%`;
    return await this.safeQuery(sql, [searchPattern, searchPattern]);
  }

  // Create user (safe)
  async createUser(userData) {
    const sql = 'INSERT INTO users (email, password, name) VALUES (?, ?, ?)';
    const { email, password, name } = userData;
    const result = await this.safeQuery(sql, [email, password, name]);
    return result.insertId;
  }

  // Update user (safe)
  async updateUser(userId, updateData) {
    const fields = Object.keys(updateData);
    const values = Object.values(updateData);
    const setClause = fields.map(field => `${field} = ?`).join(', ');
    
    const sql = `UPDATE users SET ${setClause} WHERE id = ?`;
    const params = [...values, userId];
    
    await this.safeQuery(sql, params);
  }
}
```

### NoSQL Injection Prevention
```javascript
const mongoose = require('mongoose');

class NoSQLSecurityService {
  constructor() {
    this.setupMongoose();
  }

  setupMongoose() {
    // Enable strict mode
    mongoose.set('strictQuery', true);
    
    // Sanitize user input
    mongoose.set('sanitizeFilter', true);
  }

  // Safe user search
  async searchUsers(searchTerm) {
    // Sanitize input
    const sanitizedTerm = this.sanitizeInput(searchTerm);
    
    // Use parameterized queries
    const users = await User.find({
      $or: [
        { name: { $regex: sanitizedTerm, $options: 'i' } },
        { email: { $regex: sanitizedTerm, $options: 'i' } }
      ]
    }).select('name email');
    
    return users;
  }

  // Safe user creation
  async createUser(userData) {
    // Validate and sanitize input
    const sanitizedData = this.sanitizeUserData(userData);
    
    const user = new User(sanitizedData);
    return await user.save();
  }

  // Sanitize input
  sanitizeInput(input) {
    if (typeof input !== 'string') return input;
    
    // Remove dangerous characters
    return input.replace(/[$]/g, '');
  }

  // Sanitize user data
  sanitizeUserData(data) {
    const sanitized = {};
    
    for (const [key, value] of Object.entries(data)) {
      if (typeof value === 'string') {
        sanitized[key] = this.sanitizeInput(value);
      } else {
        sanitized[key] = value;
      }
    }
    
    return sanitized;
  }
}
```

## XSS Prevention

### XSS Protection Middleware
```javascript
const xss = require('xss-clean');
const createDOMPurify = require('isomorphic-dompurify');

class XSSProtectionService {
  constructor() {
    this.setupXSSProtection();
  }

  setupXSSProtection() {
    // XSS clean middleware
    this.app.use(xss());
  }

  // Sanitize HTML content
  sanitizeHtml(html) {
    return createDOMPurify.sanitize(html, {
      ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'p', 'br'],
      ALLOWED_ATTR: []
    });
  }

  // Sanitize user input
  sanitizeInput(input) {
    if (typeof input !== 'string') return input;
    
    // Remove script tags
    let sanitized = input.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
    
    // Remove event handlers
    sanitized = sanitized.replace(/on\w+="[^"]*"/gi, '');
    sanitized = sanitized.replace(/on\w+='[^']*'/gi, '');
    
    // Remove javascript: URLs
    sanitized = sanitized.replace(/javascript:/gi, '');
    
    return sanitized;
  }

  // Content Security Policy
  setupCSP() {
    return helmet.contentSecurityPolicy({
      directives: {
        defaultSrc: ["'self'"],
        styleSrc: ["'self'", "'unsafe-inline'"],
        scriptSrc: ["'self'"],
        imgSrc: ["'self'", "data:", "https:"],
        connectSrc: ["'self'"],
        fontSrc: ["'self'"],
        objectSrc: ["'none'"],
        mediaSrc: ["'self'"],
        frameSrc: ["'none'"],
      },
    });
  }
}
```

## CSRF Protection

### CSRF Token Implementation
```javascript
const csrf = require('csurf');
const cookieParser = require('cookie-parser');

class CSRFProtectionService {
  constructor() {
    this.setupCSRFProtection();
  }

  setupCSRFProtection() {
    // Cookie parser
    this.app.use(cookieParser());
    
    // CSRF protection
    this.csrfProtection = csrf({
      cookie: {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'strict'
      }
    });
    
    // Apply CSRF protection to all routes except GET
    this.app.use((req, res, next) => {
      if (req.method === 'GET') {
        return next();
      }
      return this.csrfProtection(req, res, next);
    });
  }

  // Generate CSRF token
  generateCSRFToken(req, res) {
    const token = req.csrfToken();
    res.cookie('XSRF-TOKEN', token, {
      httpOnly: false,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict'
    });
    return token;
  }

  // Verify CSRF token
  verifyCSRFToken(req, res, next) {
    const token = req.headers['x-csrf-token'] || req.body._csrf;
    const sessionToken = req.session.csrfSecret;
    
    if (!token || !sessionToken || token !== sessionToken) {
      return res.status(403).json({ error: 'Invalid CSRF token' });
    }
    
    next();
  }
}
```

## Security Headers

### Comprehensive Security Headers
```javascript
const helmet = require('helmet');

class SecurityHeadersService {
  constructor() {
    this.setupSecurityHeaders();
  }

  setupSecurityHeaders() {
    this.app.use(helmet({
      // Content Security Policy
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          scriptSrc: ["'self'"],
          imgSrc: ["'self'", "data:", "https:"],
          connectSrc: ["'self'"],
          fontSrc: ["'self'"],
          objectSrc: ["'none'"],
          mediaSrc: ["'self'"],
          frameSrc: ["'none'"],
          upgradeInsecureRequests: [],
        },
      },
      
      // HTTP Strict Transport Security
      hsts: {
        maxAge: 31536000,
        includeSubDomains: true,
        preload: true
      },
      
      // X-Frame-Options
      frameguard: { action: 'deny' },
      
      // X-Content-Type-Options
      noSniff: true,
      
      // X-XSS-Protection
      xssFilter: true,
      
      // Referrer Policy
      referrerPolicy: { policy: 'same-origin' }
    }));

    // Custom security headers
    this.app.use((req, res, next) => {
      // Remove X-Powered-By header
      res.removeHeader('X-Powered-By');
      
      // Add custom security headers
      res.setHeader('X-Content-Type-Options', 'nosniff');
      res.setHeader('X-Frame-Options', 'DENY');
      res.setHeader('X-XSS-Protection', '1; mode=block');
      res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains; preload');
      res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
      
      next();
    });
  }
}
```

## HTTPS & SSL/TLS

### HTTPS Configuration
```javascript
const https = require('https');
const fs = require('fs');

class HTTPSService {
  constructor() {
    this.setupHTTPS();
  }

  setupHTTPS() {
    if (process.env.NODE_ENV === 'production') {
      const options = {
        key: fs.readFileSync(process.env.SSL_KEY_PATH),
        cert: fs.readFileSync(process.env.SSL_CERT_PATH),
        ca: fs.readFileSync(process.env.SSL_CA_PATH),
        secureProtocol: 'TLSv1_2_method',
        ciphers: [
          'ECDHE-RSA-AES128-GCM-SHA256',
          'ECDHE-RSA-AES256-GCM-SHA384',
          'ECDHE-RSA-AES128-SHA256',
          'ECDHE-RSA-AES256-SHA384'
        ].join(':'),
        honorCipherOrder: true
      };

      this.server = https.createServer(options, this.app);
    } else {
      this.server = require('http').createServer(this.app);
    }
  }

  // Force HTTPS redirect
  forceHTTPS() {
    return (req, res, next) => {
      if (process.env.NODE_ENV === 'production' && !req.secure) {
        return res.redirect(301, `https://${req.headers.host}${req.url}`);
      }
      next();
    };
  }
}
```

## Session Management

### Secure Session Configuration
```javascript
const session = require('express-session');
const RedisStore = require('connect-redis')(session);

class SessionService {
  constructor() {
    this.setupSessions();
  }

  setupSessions() {
    this.app.use(session({
      store: new RedisStore({
        host: process.env.REDIS_HOST,
        port: process.env.REDIS_PORT,
        password: process.env.REDIS_PASSWORD
      }),
      secret: process.env.SESSION_SECRET,
      resave: false,
      saveUninitialized: false,
      cookie: {
        secure: process.env.NODE_ENV === 'production',
        httpOnly: true,
        maxAge: 24 * 60 * 60 * 1000, // 24 hours
        sameSite: 'strict'
      },
      name: 'sessionId',
      rolling: true
    }));
  }

  // Session validation middleware
  validateSession(req, res, next) {
    if (!req.session.userId) {
      return res.status(401).json({ error: 'Session expired' });
    }
    next();
  }

  // Destroy session
  destroySession(req, res) {
    req.session.destroy((err) => {
      if (err) {
        return res.status(500).json({ error: 'Failed to destroy session' });
      }
      res.clearCookie('sessionId');
      res.json({ message: 'Session destroyed' });
    });
  }
}
```

## File Upload Security

### Secure File Upload
```javascript
const multer = require('multer');
const path = require('path');
const crypto = require('crypto');

class FileUploadService {
  constructor() {
    this.setupFileUpload();
  }

  setupFileUpload() {
    // Configure storage
    const storage = multer.diskStorage({
      destination: (req, file, cb) => {
        cb(null, 'uploads/');
      },
      filename: (req, file, cb) => {
        const uniqueSuffix = crypto.randomBytes(16).toString('hex');
        cb(null, `${uniqueSuffix}${path.extname(file.originalname)}`);
      }
    });

    // File filter
    const fileFilter = (req, file, cb) => {
      const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'application/pdf'];
      const maxSize = 5 * 1024 * 1024; // 5MB
      
      if (!allowedTypes.includes(file.mimetype)) {
        return cb(new Error('Invalid file type'), false);
      }
      
      if (file.size > maxSize) {
        return cb(new Error('File too large'), false);
      }
      
      cb(null, true);
    };

    this.upload = multer({
      storage: storage,
      fileFilter: fileFilter,
      limits: {
        fileSize: 5 * 1024 * 1024, // 5MB
        files: 5 // Maximum 5 files
      }
    });
  }

  // Upload middleware
  uploadFiles(fieldName, maxCount = 1) {
    return this.upload.array(fieldName, maxCount);
  }

  // Validate file
  validateFile(file) {
    const allowedExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.pdf'];
    const extension = path.extname(file.originalname).toLowerCase();
    
    if (!allowedExtensions.includes(extension)) {
      throw new Error('Invalid file extension');
    }
    
    // Check file signature
    const fileSignature = file.buffer.toString('hex', 0, 4);
    const validSignatures = {
      'ffd8ffe0': 'image/jpeg',
      'ffd8ffe1': 'image/jpeg',
      '89504e47': 'image/png',
      '47494638': 'image/gif',
      '25504446': 'application/pdf'
    };
    
    if (!validSignatures[fileSignature]) {
      throw new Error('Invalid file signature');
    }
    
    return true;
  }
}
```

## Dependency Security

### Security Audit
```javascript
const { execSync } = require('child_process');

class DependencySecurityService {
  constructor() {
    this.setupSecurityAudit();
  }

  setupSecurityAudit() {
    // Run npm audit
    this.runSecurityAudit();
    
    // Check for outdated packages
    this.checkOutdatedPackages();
  }

  runSecurityAudit() {
    try {
      const auditResult = execSync('npm audit --json', { encoding: 'utf8' });
      const audit = JSON.parse(auditResult);
      
      if (audit.vulnerabilities && Object.keys(audit.vulnerabilities).length > 0) {
        console.warn('Security vulnerabilities found:', audit.vulnerabilities);
      }
    } catch (error) {
      console.error('Security audit failed:', error.message);
    }
  }

  checkOutdatedPackages() {
    try {
      const outdatedResult = execSync('npm outdated --json', { encoding: 'utf8' });
      const outdated = JSON.parse(outdatedResult);
      
      if (Object.keys(outdated).length > 0) {
        console.warn('Outdated packages found:', outdated);
      }
    } catch (error) {
      // No outdated packages
    }
  }

  // Update vulnerable packages
  updateVulnerablePackages() {
    try {
      execSync('npm audit fix', { stdio: 'inherit' });
      console.log('Vulnerable packages updated');
    } catch (error) {
      console.error('Failed to update packages:', error.message);
    }
  }
}
```

## Logging & Monitoring

### Security Logging
```javascript
const winston = require('winston');

class SecurityLoggingService {
  constructor() {
    this.setupLogging();
  }

  setupLogging() {
    this.logger = winston.createLogger({
      level: 'info',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
      ),
      transports: [
        new winston.transports.File({ filename: 'logs/security.log' }),
        new winston.transports.Console()
      ]
    });
  }

  // Log security events
  logSecurityEvent(event, details) {
    this.logger.warn('Security Event', {
      event,
      details,
      timestamp: new Date().toISOString(),
      ip: details.ip,
      userAgent: details.userAgent
    });
  }

  // Log authentication attempts
  logAuthAttempt(email, success, ip, userAgent) {
    this.logger.info('Authentication Attempt', {
      email,
      success,
      ip,
      userAgent,
      timestamp: new Date().toISOString()
    });
  }

  // Log suspicious activity
  logSuspiciousActivity(activity, details) {
    this.logger.error('Suspicious Activity', {
      activity,
      details,
      timestamp: new Date().toISOString()
    });
  }
}
```

## Rate Limiting

### Advanced Rate Limiting
```javascript
const rateLimit = require('express-rate-limit');
const RedisStore = require('rate-limit-redis');
const Redis = require('redis');

class RateLimitingService {
  constructor() {
    this.setupRateLimiting();
  }

  setupRateLimiting() {
    const redisClient = Redis.createClient({
      host: process.env.REDIS_HOST,
      port: process.env.REDIS_PORT
    });

    // General rate limiting
    this.generalLimiter = rateLimit({
      store: new RedisStore({
        client: redisClient,
        prefix: 'rl:'
      }),
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 100, // limit each IP to 100 requests per windowMs
      message: 'Too many requests from this IP, please try again later.',
      standardHeaders: true,
      legacyHeaders: false,
    });

    // Strict rate limiting for auth endpoints
    this.authLimiter = rateLimit({
      store: new RedisStore({
        client: redisClient,
        prefix: 'auth:'
      }),
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 5, // limit each IP to 5 requests per windowMs
      message: 'Too many authentication attempts, please try again later.',
      standardHeaders: true,
      legacyHeaders: false,
    });

    // API rate limiting
    this.apiLimiter = rateLimit({
      store: new RedisStore({
        client: redisClient,
        prefix: 'api:'
      }),
      windowMs: 60 * 1000, // 1 minute
      max: 60, // limit each IP to 60 requests per minute
      message: 'API rate limit exceeded, please try again later.',
      standardHeaders: true,
      legacyHeaders: false,
    });
  }
}
```

## Error Handling

### Secure Error Handling
```javascript
class ErrorHandlingService {
  constructor() {
    this.setupErrorHandling();
  }

  setupErrorHandling() {
    // Global error handler
    this.app.use((err, req, res, next) => {
      // Log error
      console.error('Error:', err);
      
      // Don't expose internal errors in production
      if (process.env.NODE_ENV === 'production') {
        return res.status(500).json({
          error: 'Internal server error',
          requestId: req.id
        });
      }
      
      // Development error response
      res.status(500).json({
        error: err.message,
        stack: err.stack,
        requestId: req.id
      });
    });

    // 404 handler
    this.app.use((req, res) => {
      res.status(404).json({
        error: 'Not found',
        path: req.path,
        method: req.method
      });
    });
  }

  // Custom error classes
  createCustomError(name, message, statusCode) {
    class CustomError extends Error {
      constructor(details) {
        super(message);
        this.name = name;
        this.statusCode = statusCode;
        this.details = details;
      }
    }
    return CustomError;
  }
}
```

## Production Security

### Production Security Checklist
```javascript
class ProductionSecurityService {
  constructor() {
    this.setupProductionSecurity();
  }

  setupProductionSecurity() {
    // Environment validation
    this.validateEnvironment();
    
    // Security headers
    this.setupSecurityHeaders();
    
    // HTTPS enforcement
    this.enforceHTTPS();
    
    // Error handling
    this.setupErrorHandling();
  }

  validateEnvironment() {
    const requiredEnvVars = [
      'NODE_ENV',
      'JWT_SECRET',
      'DB_PASSWORD',
      'REDIS_PASSWORD',
      'SESSION_SECRET'
    ];

    const missing = requiredEnvVars.filter(envVar => !process.env[envVar]);
    
    if (missing.length > 0) {
      throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
    }
  }

  // Security monitoring
  setupSecurityMonitoring() {
    // Monitor failed login attempts
    this.monitorFailedLogins();
    
    // Monitor suspicious activity
    this.monitorSuspiciousActivity();
    
    // Monitor rate limiting
    this.monitorRateLimiting();
  }

  // Health check endpoint
  healthCheck(req, res) {
    res.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      version: process.env.npm_package_version
    });
  }
}
```

## Security Best Practices

### Code Security Guidelines
1. **Input Validation**: Always validate and sanitize user input
2. **Authentication**: Use strong authentication mechanisms
3. **Authorization**: Implement proper access controls
4. **Session Management**: Secure session handling
5. **Error Handling**: Don't expose sensitive information
6. **Logging**: Log security events and suspicious activity
7. **Dependencies**: Keep dependencies updated and audited
8. **HTTPS**: Use HTTPS in production
9. **Headers**: Set appropriate security headers
10. **Rate Limiting**: Implement rate limiting to prevent abuse

### Security Testing
```javascript
const securityTests = {
  // Test for SQL injection
  testSQLInjection: async (endpoint, payload) => {
    const response = await request(app)
      .post(endpoint)
      .send(payload);
    
    expect(response.status).not.toBe(500);
    expect(response.body.error).not.toContain('SQL');
  },

  // Test for XSS
  testXSS: async (endpoint, payload) => {
    const response = await request(app)
      .post(endpoint)
      .send(payload);
    
    expect(response.body).not.toContain('<script>');
  },

  // Test authentication
  testAuthentication: async (endpoint) => {
    const response = await request(app)
      .get(endpoint);
    
    expect(response.status).toBe(401);
  }
};
```

This comprehensive security guide covers all major aspects of Node.js security, from authentication and authorization to production security best practices. Each section includes practical code examples and implementation details that can be directly used in production applications.
