import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';
import rateLimit from 'express-rate-limit';
import { createServer } from 'http';
import { Server as SocketIOServer } from 'socket.io';
import dotenv from 'dotenv';

// Import routes
import parkingRoutes from './api/parking/parkingRoutes';
import tradingRoutes from './api/trading/tradingRoutes';
import userRoutes from './api/user/userRoutes';

// Import services
import { DatabaseService } from './services/database/DatabaseService';
import { RedisService } from './services/cache/RedisService';
import { KafkaService } from './services/messaging/KafkaService';
import { WebSocketService } from './services/websocket/WebSocketService';
import { Logger } from './utils/Logger';

// Load environment variables
dotenv.config();

class TradingPlatformApp {
  private app: express.Application;
  private server: any;
  private io: SocketIOServer;
  private logger: Logger;
  private databaseService: DatabaseService;
  private redisService: RedisService;
  private kafkaService: KafkaService;
  private webSocketService: WebSocketService;

  constructor() {
    this.app = express();
    this.server = createServer(this.app);
    this.io = new SocketIOServer(this.server, {
      cors: {
        origin: process.env.CORS_ORIGIN || "http://localhost:3000",
        methods: ["GET", "POST"]
      }
    });
    
    this.logger = new Logger('TradingPlatformApp');
    this.databaseService = new DatabaseService();
    this.redisService = new RedisService();
    this.kafkaService = new KafkaService();
    this.webSocketService = new WebSocketService(this.io);
  }

  /**
   * Initialize the application
   */
  async initialize(): Promise<void> {
    try {
      this.logger.info('Initializing Trading Platform Application...');

      // Setup middleware
      this.setupMiddleware();

      // Setup routes
      this.setupRoutes();

      // Setup error handling
      this.setupErrorHandling();

      // Initialize services
      await this.initializeServices();

      // Setup WebSocket
      this.setupWebSocket();

      this.logger.info('Application initialized successfully');
    } catch (error) {
      this.logger.error('Failed to initialize application', error);
      throw error;
    }
  }

  /**
   * Setup middleware
   */
  private setupMiddleware(): void {
    // Security middleware
    this.app.use(helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          scriptSrc: ["'self'"],
          imgSrc: ["'self'", "data:", "https:"],
        },
      },
    }));

    // CORS
    this.app.use(cors({
      origin: process.env.CORS_ORIGIN || "http://localhost:3000",
      credentials: true,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization', 'x-api-key']
    }));

    // Compression
    this.app.use(compression());

    // Logging
    this.app.use(morgan('combined', {
      stream: {
        write: (message: string) => this.logger.info(message.trim())
      }
    }));

    // Rate limiting
    const limiter = rateLimit({
      windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS || '900000'), // 15 minutes
      max: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS || '100'),
      message: {
        success: false,
        message: 'Too many requests from this IP, please try again later.'
      },
      standardHeaders: true,
      legacyHeaders: false,
    });
    this.app.use('/api/', limiter);

    // Body parsing
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));

    // Request ID middleware
    this.app.use((req, res, next) => {
      req.id = this.generateRequestId();
      res.setHeader('X-Request-ID', req.id);
      next();
    });
  }

  /**
   * Setup routes
   */
  private setupRoutes(): void {
    // Health check
    this.app.get('/health', (req, res) => {
      res.status(200).json({
        success: true,
        message: 'Trading Platform is healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        version: process.env.npm_package_version || '1.0.0'
      });
    });

    // API routes
    this.app.use('/api/v1/parking', parkingRoutes);
    this.app.use('/api/v1/trading', tradingRoutes);
    this.app.use('/api/v1/users', userRoutes);

    // API documentation
    this.app.get('/api/docs', (req, res) => {
      res.json({
        message: 'Trading Platform API Documentation',
        version: '1.0.0',
        endpoints: {
          parking: '/api/v1/parking',
          trading: '/api/v1/trading',
          users: '/api/v1/users'
        }
      });
    });

    // 404 handler
    this.app.use('*', (req, res) => {
      res.status(404).json({
        success: false,
        message: 'Endpoint not found',
        path: req.originalUrl
      });
    });
  }

  /**
   * Setup error handling
   */
  private setupErrorHandling(): void {
    // Global error handler
    this.app.use((error: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
      this.logger.error('Unhandled error', {
        error: error.message,
        stack: error.stack,
        requestId: req.id,
        path: req.path,
        method: req.method
      });

      const statusCode = error.statusCode || 500;
      const message = process.env.NODE_ENV === 'production' 
        ? 'Internal server error' 
        : error.message;

      res.status(statusCode).json({
        success: false,
        message,
        requestId: req.id,
        ...(process.env.NODE_ENV === 'development' && { stack: error.stack })
      });
    });

    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
      this.logger.error('Uncaught Exception', error);
      process.exit(1);
    });

    // Handle unhandled promise rejections
    process.on('unhandledRejection', (reason, promise) => {
      this.logger.error('Unhandled Rejection', { reason, promise });
      process.exit(1);
    });

    // Graceful shutdown
    process.on('SIGTERM', () => {
      this.logger.info('SIGTERM received, shutting down gracefully');
      this.shutdown();
    });

    process.on('SIGINT', () => {
      this.logger.info('SIGINT received, shutting down gracefully');
      this.shutdown();
    });
  }

  /**
   * Initialize services
   */
  private async initializeServices(): Promise<void> {
    try {
      // Initialize database
      await this.databaseService.initialize();
      this.logger.info('Database service initialized');

      // Initialize Redis
      await this.redisService.initialize();
      this.logger.info('Redis service initialized');

      // Initialize Kafka
      await this.kafkaService.initialize();
      this.logger.info('Kafka service initialized');

      // Initialize WebSocket service
      await this.webSocketService.initialize();
      this.logger.info('WebSocket service initialized');

    } catch (error) {
      this.logger.error('Failed to initialize services', error);
      throw error;
    }
  }

  /**
   * Setup WebSocket
   */
  private setupWebSocket(): void {
    this.io.on('connection', (socket) => {
      this.logger.info('Client connected', { socketId: socket.id });

      // Handle parking events
      socket.on('join_parking_lot', (lotId: string) => {
        socket.join(`parking_lot_${lotId}`);
        this.logger.info('Client joined parking lot', { socketId: socket.id, lotId });
      });

      // Handle trading events
      socket.on('join_trading_room', (userId: string) => {
        socket.join(`trading_room_${userId}`);
        this.logger.info('Client joined trading room', { socketId: socket.id, userId });
      });

      // Handle disconnection
      socket.on('disconnect', (reason) => {
        this.logger.info('Client disconnected', { socketId: socket.id, reason });
      });

      // Handle errors
      socket.on('error', (error) => {
        this.logger.error('Socket error', { socketId: socket.id, error });
      });
    });
  }

  /**
   * Start the server
   */
  async start(): Promise<void> {
    try {
      const port = process.env.PORT || 3000;
      const wsPort = process.env.WS_PORT || 3001;

      await this.initialize();

      this.server.listen(port, () => {
        this.logger.info(`Trading Platform server started on port ${port}`);
        this.logger.info(`WebSocket server started on port ${wsPort}`);
        this.logger.info(`Environment: ${process.env.NODE_ENV || 'development'}`);
      });

    } catch (error) {
      this.logger.error('Failed to start server', error);
      throw error;
    }
  }

  /**
   * Shutdown the application gracefully
   */
  private async shutdown(): Promise<void> {
    try {
      this.logger.info('Starting graceful shutdown...');

      // Close server
      this.server.close(() => {
        this.logger.info('HTTP server closed');
      });

      // Close WebSocket
      this.io.close(() => {
        this.logger.info('WebSocket server closed');
      });

      // Close services
      await this.databaseService.close();
      await this.redisService.close();
      await this.kafkaService.close();

      this.logger.info('Graceful shutdown completed');
      process.exit(0);
    } catch (error) {
      this.logger.error('Error during shutdown', error);
      process.exit(1);
    }
  }

  /**
   * Generate unique request ID
   */
  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

// Start the application
const app = new TradingPlatformApp();
app.start().catch((error) => {
  console.error('Failed to start application:', error);
  process.exit(1);
});

export default app;
