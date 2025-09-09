import { Request, Response } from 'express';
import { ParkingService } from '../../services/parking/ParkingService';
import { Logger } from '../../utils/Logger';
import { validateRequest } from '../../middleware/validation';
import { authenticateToken } from '../../middleware/auth';
import { rateLimit } from '../../middleware/rateLimit';

export class ParkingController {
  private parkingService: ParkingService;
  private logger: Logger;

  constructor() {
    this.parkingService = new ParkingService();
    this.logger = new Logger('ParkingController');
  }

  /**
   * Get all parking lots
   * GET /api/v1/parking/lots
   */
  getParkingLots = async (req: Request, res: Response): Promise<void> => {
    try {
      const lots = await this.parkingService.getParkingLots();
      
      res.status(200).json({
        success: true,
        data: lots,
        count: lots.length
      });
    } catch (error) {
      this.logger.error('Failed to get parking lots', error);
      res.status(500).json({
        success: false,
        message: 'Failed to retrieve parking lots',
        error: process.env.NODE_ENV === 'development' ? error : undefined
      });
    }
  };

  /**
   * Get parking lot by ID
   * GET /api/v1/parking/lots/:id
   */
  getParkingLotById = async (req: Request, res: Response): Promise<void> => {
    try {
      const { id } = req.params;
      
      if (!id) {
        res.status(400).json({
          success: false,
          message: 'Parking lot ID is required'
        });
        return;
      }

      const lot = await this.parkingService.getParkingLotById(id);
      
      if (!lot) {
        res.status(404).json({
          success: false,
          message: 'Parking lot not found'
        });
        return;
      }

      res.status(200).json({
        success: true,
        data: lot
      });
    } catch (error) {
      this.logger.error('Failed to get parking lot by ID', { lotId: req.params.id, error });
      res.status(500).json({
        success: false,
        message: 'Failed to retrieve parking lot',
        error: process.env.NODE_ENV === 'development' ? error : undefined
      });
    }
  };

  /**
   * Get available spaces in a parking lot
   * GET /api/v1/parking/lots/:id/spaces/available
   */
  getAvailableSpaces = async (req: Request, res: Response): Promise<void> => {
    try {
      const { id: lotId } = req.params;
      const { 
        userType = 'visitor',
        startTime,
        endTime,
        spaceType,
        floor,
        features,
        accessibility
      } = req.query;

      if (!lotId) {
        res.status(400).json({
          success: false,
          message: 'Parking lot ID is required'
        });
        return;
      }

      // Parse dates
      const start = startTime ? new Date(startTime as string) : new Date();
      const end = endTime ? new Date(endTime as string) : new Date(Date.now() + 2 * 60 * 60 * 1000); // Default 2 hours

      if (isNaN(start.getTime()) || isNaN(end.getTime())) {
        res.status(400).json({
          success: false,
          message: 'Invalid date format'
        });
        return;
      }

      if (start >= end) {
        res.status(400).json({
          success: false,
          message: 'End time must be after start time'
        });
        return;
      }

      const preferences = {
        spaceType: spaceType ? (spaceType as string).split(',') : undefined,
        floor: floor ? (floor as string).split(',').map(Number) : undefined,
        features: features ? (features as string).split(',') : undefined,
        accessibility: accessibility === 'true'
      };

      const spaces = await this.parkingService.getAvailableSpaces(
        lotId,
        userType as string,
        start,
        end,
        preferences
      );

      res.status(200).json({
        success: true,
        data: spaces,
        count: spaces.length,
        filters: {
          lotId,
          userType,
          startTime: start,
          endTime: end,
          preferences
        }
      });
    } catch (error) {
      this.logger.error('Failed to get available spaces', { 
        lotId: req.params.id, 
        query: req.query, 
        error 
      });
      res.status(500).json({
        success: false,
        message: 'Failed to retrieve available spaces',
        error: process.env.NODE_ENV === 'development' ? error : undefined
      });
    }
  };

  /**
   * Create a new booking
   * POST /api/v1/parking/bookings
   */
  createBooking = [
    authenticateToken,
    rateLimit({ windowMs: 15 * 60 * 1000, max: 10 }), // 10 requests per 15 minutes
    validateRequest({
      body: {
        lotId: { type: 'string', required: true },
        spaceId: { type: 'string', required: true },
        startTime: { type: 'string', required: true },
        endTime: { type: 'string', required: true },
        vehicleInfo: {
          type: 'object',
          required: true,
          properties: {
            licensePlate: { type: 'string', required: true },
            make: { type: 'string', required: true },
            model: { type: 'string', required: true },
            color: { type: 'string', required: true },
            type: { type: 'string', required: true }
          }
        }
      }
    }),
    async (req: Request, res: Response): Promise<void> => {
      try {
        const { lotId, spaceId, startTime, endTime, vehicleInfo } = req.body;
        const userId = req.user?.id;

        if (!userId) {
          res.status(401).json({
            success: false,
            message: 'User not authenticated'
          });
          return;
        }

        // Parse dates
        const start = new Date(startTime);
        const end = new Date(endTime);

        if (isNaN(start.getTime()) || isNaN(end.getTime())) {
          res.status(400).json({
            success: false,
            message: 'Invalid date format'
          });
          return;
        }

        if (start >= end) {
          res.status(400).json({
            success: false,
            message: 'End time must be after start time'
          });
          return;
        }

        if (start < new Date()) {
          res.status(400).json({
            success: false,
            message: 'Start time cannot be in the past'
          });
          return;
        }

        const booking = await this.parkingService.createBooking(
          userId,
          lotId,
          spaceId,
          start,
          end,
          vehicleInfo
        );

        res.status(201).json({
          success: true,
          data: booking,
          message: 'Booking created successfully'
        });
      } catch (error) {
        this.logger.error('Failed to create booking', { 
          userId: req.user?.id, 
          body: req.body, 
          error 
        });
        
        const statusCode = error.message.includes('not found') ? 404 :
                          error.message.includes('not available') ? 409 :
                          error.message.includes('validation') ? 400 : 500;

        res.status(statusCode).json({
          success: false,
          message: error.message || 'Failed to create booking',
          error: process.env.NODE_ENV === 'development' ? error : undefined
        });
      }
    }
  ];

  /**
   * Get user's bookings
   * GET /api/v1/parking/bookings
   */
  getUserBookings = [
    authenticateToken,
    async (req: Request, res: Response): Promise<void> => {
      try {
        const userId = req.user?.id;
        const { limit = 10, offset = 0, status } = req.query;

        if (!userId) {
          res.status(401).json({
            success: false,
            message: 'User not authenticated'
          });
          return;
        }

        const bookings = await this.parkingService.getUserBookings(
          userId,
          parseInt(limit as string),
          parseInt(offset as string)
        );

        res.status(200).json({
          success: true,
          data: bookings,
          count: bookings.length,
          pagination: {
            limit: parseInt(limit as string),
            offset: parseInt(offset as string)
          }
        });
      } catch (error) {
        this.logger.error('Failed to get user bookings', { 
          userId: req.user?.id, 
          error 
        });
        res.status(500).json({
          success: false,
          message: 'Failed to retrieve bookings',
          error: process.env.NODE_ENV === 'development' ? error : undefined
        });
      }
    }
  ];

  /**
   * Get booking by ID
   * GET /api/v1/parking/bookings/:id
   */
  getBookingById = [
    authenticateToken,
    async (req: Request, res: Response): Promise<void> => {
      try {
        const { id } = req.params;
        const userId = req.user?.id;

        if (!id) {
          res.status(400).json({
            success: false,
            message: 'Booking ID is required'
          });
          return;
        }

        // Get booking from service (includes authorization check)
        const booking = await this.parkingService.getBookingById(id, userId);

        if (!booking) {
          res.status(404).json({
            success: false,
            message: 'Booking not found'
          });
          return;
        }

        res.status(200).json({
          success: true,
          data: booking
        });
      } catch (error) {
        this.logger.error('Failed to get booking by ID', { 
          bookingId: req.params.id, 
          userId: req.user?.id, 
          error 
        });
        res.status(500).json({
          success: false,
          message: 'Failed to retrieve booking',
          error: process.env.NODE_ENV === 'development' ? error : undefined
        });
      }
    }
  ];

  /**
   * Cancel a booking
   * DELETE /api/v1/parking/bookings/:id
   */
  cancelBooking = [
    authenticateToken,
    rateLimit({ windowMs: 15 * 60 * 1000, max: 5 }), // 5 cancellations per 15 minutes
    async (req: Request, res: Response): Promise<void> => {
      try {
        const { id } = req.params;
        const userId = req.user?.id;

        if (!id) {
          res.status(400).json({
            success: false,
            message: 'Booking ID is required'
          });
          return;
        }

        if (!userId) {
          res.status(401).json({
            success: false,
            message: 'User not authenticated'
          });
          return;
        }

        const result = await this.parkingService.cancelBooking(id, userId);

        res.status(200).json({
          success: true,
          data: result,
          message: 'Booking cancelled successfully'
        });
      } catch (error) {
        this.logger.error('Failed to cancel booking', { 
          bookingId: req.params.id, 
          userId: req.user?.id, 
          error 
        });
        
        const statusCode = error.message.includes('not found') ? 404 :
                          error.message.includes('Unauthorized') ? 403 :
                          error.message.includes('Cannot cancel') ? 400 : 500;

        res.status(statusCode).json({
          success: false,
          message: error.message || 'Failed to cancel booking',
          error: process.env.NODE_ENV === 'development' ? error : undefined
        });
      }
    }
  ];

  /**
   * Process entry (check-in)
   * POST /api/v1/parking/entry
   */
  processEntry = [
    rateLimit({ windowMs: 60 * 1000, max: 100 }), // 100 requests per minute
    validateRequest({
      body: {
        rfidTag: { type: 'string', required: true },
        gateId: { type: 'string', required: true }
      }
    }),
    async (req: Request, res: Response): Promise<void> => {
      try {
        const { rfidTag, gateId } = req.body;

        const result = await this.parkingService.processEntry(rfidTag, gateId);

        const statusCode = result.success ? 200 : 403;
        res.status(statusCode).json({
          success: result.success,
          message: result.message,
          data: result.bookingId ? { bookingId: result.bookingId } : undefined
        });
      } catch (error) {
        this.logger.error('Failed to process entry', { 
          body: req.body, 
          error 
        });
        res.status(500).json({
          success: false,
          message: 'Failed to process entry',
          error: process.env.NODE_ENV === 'development' ? error : undefined
        });
      }
    }
  ];

  /**
   * Process exit (check-out)
   * POST /api/v1/parking/exit
   */
  processExit = [
    rateLimit({ windowMs: 60 * 1000, max: 100 }), // 100 requests per minute
    validateRequest({
      body: {
        rfidTag: { type: 'string', required: true },
        gateId: { type: 'string', required: true }
      }
    }),
    async (req: Request, res: Response): Promise<void> => {
      try {
        const { rfidTag, gateId } = req.body;

        const result = await this.parkingService.processExit(rfidTag, gateId);

        const statusCode = result.success ? 200 : 403;
        res.status(statusCode).json({
          success: result.success,
          message: result.message,
          data: result.finalAmount ? { finalAmount: result.finalAmount } : undefined
        });
      } catch (error) {
        this.logger.error('Failed to process exit', { 
          body: req.body, 
          error 
        });
        res.status(500).json({
          success: false,
          message: 'Failed to process exit',
          error: process.env.NODE_ENV === 'development' ? error : undefined
        });
      }
    }
  ];

  /**
   * Get parking status for a lot
   * GET /api/v1/parking/lots/:id/status
   */
  getParkingStatus = async (req: Request, res: Response): Promise<void> => {
    try {
      const { id: lotId } = req.params;

      if (!lotId) {
        res.status(400).json({
          success: false,
          message: 'Parking lot ID is required'
        });
        return;
      }

      const status = await this.parkingService.getParkingStatus(lotId);

      res.status(200).json({
        success: true,
        data: status
      });
    } catch (error) {
      this.logger.error('Failed to get parking status', { 
        lotId: req.params.id, 
        error 
      });
      res.status(500).json({
        success: false,
        message: 'Failed to retrieve parking status',
        error: process.env.NODE_ENV === 'development' ? error : undefined
      });
    }
  };

  /**
   * Health check endpoint
   * GET /api/v1/parking/health
   */
  healthCheck = async (req: Request, res: Response): Promise<void> => {
    try {
      // Check database connection
      const dbStatus = await this.parkingService.healthCheck();
      
      res.status(200).json({
        success: true,
        data: {
          status: 'healthy',
          timestamp: new Date().toISOString(),
          services: {
            database: dbStatus.database ? 'connected' : 'disconnected',
            cache: dbStatus.cache ? 'connected' : 'disconnected',
            websocket: dbStatus.websocket ? 'connected' : 'disconnected'
          }
        }
      });
    } catch (error) {
      this.logger.error('Health check failed', error);
      res.status(503).json({
        success: false,
        data: {
          status: 'unhealthy',
          timestamp: new Date().toISOString(),
          error: error.message
        }
      });
    }
  };
}
