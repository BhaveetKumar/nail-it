import { ParkingLot, ParkingSpace, Booking, User, VehicleInfo } from '../../models/ParkingModels';
import { SpaceAllocationService } from './SpaceAllocationService';
import { PricingService } from './PricingService';
import { AccessControlService } from './AccessControlService';
import { PaymentService } from './PaymentService';
import { NotificationService } from './NotificationService';
import { DatabaseService } from '../database/DatabaseService';
import { RedisService } from '../cache/RedisService';
import { WebSocketService } from '../websocket/WebSocketService';
import { Logger } from '../../utils/Logger';

export class ParkingService {
  private spaceAllocationService: SpaceAllocationService;
  private pricingService: PricingService;
  private accessControlService: AccessControlService;
  private paymentService: PaymentService;
  private notificationService: NotificationService;
  private databaseService: DatabaseService;
  private redisService: RedisService;
  private webSocketService: WebSocketService;
  private logger: Logger;

  constructor() {
    this.spaceAllocationService = new SpaceAllocationService();
    this.pricingService = new PricingService();
    this.accessControlService = new AccessControlService();
    this.paymentService = new PaymentService();
    this.notificationService = new NotificationService();
    this.databaseService = new DatabaseService();
    this.redisService = new RedisService();
    this.webSocketService = new WebSocketService();
    this.logger = new Logger('ParkingService');
  }

  /**
   * Get all available parking lots
   */
  async getParkingLots(): Promise<ParkingLot[]> {
    try {
      const lots = await this.databaseService.getParkingLots();
      this.logger.info('Retrieved parking lots', { count: lots.length });
      return lots;
    } catch (error) {
      this.logger.error('Failed to get parking lots', error);
      throw error;
    }
  }

  /**
   * Get parking lot by ID
   */
  async getParkingLotById(lotId: string): Promise<ParkingLot | null> {
    try {
      // Try cache first
      const cached = await this.redisService.get(`parking_lot:${lotId}`);
      if (cached) {
        return JSON.parse(cached);
      }

      const lot = await this.databaseService.getParkingLotById(lotId);
      if (lot) {
        // Cache for 5 minutes
        await this.redisService.setex(`parking_lot:${lotId}`, 300, JSON.stringify(lot));
      }

      return lot;
    } catch (error) {
      this.logger.error('Failed to get parking lot by ID', { lotId, error });
      throw error;
    }
  }

  /**
   * Get available spaces in a parking lot
   */
  async getAvailableSpaces(
    lotId: string,
    userType: string,
    startTime: Date,
    endTime: Date
  ): Promise<ParkingSpace[]> {
    try {
      const cacheKey = `available_spaces:${lotId}:${userType}:${startTime.getTime()}:${endTime.getTime()}`;
      
      // Try cache first
      const cached = await this.redisService.get(cacheKey);
      if (cached) {
        return JSON.parse(cached);
      }

      const spaces = await this.spaceAllocationService.getAvailableSpaces(
        lotId,
        userType,
        startTime,
        endTime
      );

      // Cache for 1 minute
      await this.redisService.setex(cacheKey, 60, JSON.stringify(spaces));

      this.logger.info('Retrieved available spaces', { 
        lotId, 
        userType, 
        count: spaces.length 
      });

      return spaces;
    } catch (error) {
      this.logger.error('Failed to get available spaces', { lotId, userType, error });
      throw error;
    }
  }

  /**
   * Create a new booking
   */
  async createBooking(
    userId: string,
    lotId: string,
    spaceId: string,
    startTime: Date,
    endTime: Date,
    vehicleInfo: VehicleInfo
  ): Promise<Booking> {
    try {
      // Validate user
      const user = await this.databaseService.getUserById(userId);
      if (!user) {
        throw new Error('User not found');
      }

      // Validate space availability
      const space = await this.databaseService.getSpaceById(spaceId);
      if (!space || space.status !== 'available') {
        throw new Error('Space not available');
      }

      // Calculate pricing
      const totalAmount = this.pricingService.calculatePrice(
        space,
        startTime,
        endTime,
        user.userType
      );

      // Create booking
      const booking: Booking = {
        id: this.generateBookingId(),
        userId,
        spaceId,
        lotId,
        startTime,
        endTime,
        status: 'confirmed',
        paymentId: '',
        totalAmount,
        vehicleInfo,
        createdAt: new Date(),
        updatedAt: new Date()
      };

      // Process payment
      const payment = await this.paymentService.processPayment({
        amount: totalAmount,
        userId,
        bookingId: booking.id,
        method: 'card' // Default method
      });

      booking.paymentId = payment.id;

      // Save booking to database
      await this.databaseService.createBooking(booking);

      // Update space status
      await this.databaseService.updateSpaceStatus(spaceId, 'reserved');

      // Cache invalidation
      await this.redisService.del(`available_spaces:${lotId}:*`);

      // Send notifications
      await this.notificationService.sendBookingConfirmation(user, booking);

      // WebSocket notification
      await this.webSocketService.broadcast('booking_created', {
        bookingId: booking.id,
        spaceId: booking.spaceId,
        userId: booking.userId,
        startTime: booking.startTime,
        endTime: booking.endTime
      });

      this.logger.info('Booking created successfully', { 
        bookingId: booking.id, 
        userId, 
        spaceId 
      });

      return booking;
    } catch (error) {
      this.logger.error('Failed to create booking', { 
        userId, 
        lotId, 
        spaceId, 
        error 
      });
      throw error;
    }
  }

  /**
   * Process entry (check-in)
   */
  async processEntry(rfidTag: string, gateId: string): Promise<{ success: boolean; message: string; bookingId?: string }> {
    try {
      const result = await this.accessControlService.processEntry(rfidTag, gateId);
      
      if (result.allowed) {
        // Update booking status
        await this.databaseService.updateBookingStatus(result.bookingId!, 'active');
        
        // Record check-in time
        await this.databaseService.recordCheckIn(result.bookingId!, new Date());

        // WebSocket notification
        await this.webSocketService.broadcast('check_in', {
          bookingId: result.bookingId,
          gateId,
          timestamp: new Date()
        });

        this.logger.info('Entry processed successfully', { 
          rfidTag, 
          gateId, 
          bookingId: result.bookingId 
        });

        return {
          success: true,
          message: 'Entry granted',
          bookingId: result.bookingId
        };
      } else {
        this.logger.warn('Entry denied', { rfidTag, gateId, reason: result.reason });
        return {
          success: false,
          message: result.reason
        };
      }
    } catch (error) {
      this.logger.error('Failed to process entry', { rfidTag, gateId, error });
      throw error;
    }
  }

  /**
   * Process exit (check-out)
   */
  async processExit(rfidTag: string, gateId: string): Promise<{ success: boolean; message: string; finalAmount?: number }> {
    try {
      const result = await this.accessControlService.processExit(rfidTag, gateId);
      
      if (result.allowed) {
        const booking = await this.databaseService.getBookingById(result.bookingId!);
        if (!booking) {
          throw new Error('Booking not found');
        }

        // Calculate final amount (may differ from initial estimate)
        const finalAmount = this.pricingService.calculateFinalPrice(booking);
        
        // Update booking
        await this.databaseService.updateBookingStatus(result.bookingId!, 'completed');
        await this.databaseService.recordCheckOut(result.bookingId!, new Date());

        // Update space status
        await this.databaseService.updateSpaceStatus(booking.spaceId, 'available');

        // Process any additional payment if needed
        if (finalAmount > booking.totalAmount) {
          await this.paymentService.processAdditionalPayment(booking.paymentId, finalAmount - booking.totalAmount);
        }

        // WebSocket notification
        await this.webSocketService.broadcast('check_out', {
          bookingId: result.bookingId,
          gateId,
          finalAmount,
          timestamp: new Date()
        });

        this.logger.info('Exit processed successfully', { 
          rfidTag, 
          gateId, 
          bookingId: result.bookingId,
          finalAmount
        });

        return {
          success: true,
          message: 'Exit processed',
          finalAmount
        };
      } else {
        this.logger.warn('Exit denied', { rfidTag, gateId, reason: result.reason });
        return {
          success: false,
          message: result.reason
        };
      }
    } catch (error) {
      this.logger.error('Failed to process exit', { rfidTag, gateId, error });
      throw error;
    }
  }

  /**
   * Cancel a booking
   */
  async cancelBooking(bookingId: string, userId: string): Promise<{ success: boolean; refundAmount?: number }> {
    try {
      const booking = await this.databaseService.getBookingById(bookingId);
      if (!booking) {
        throw new Error('Booking not found');
      }

      if (booking.userId !== userId) {
        throw new Error('Unauthorized to cancel this booking');
      }

      if (booking.status === 'completed' || booking.status === 'cancelled') {
        throw new Error('Cannot cancel completed or already cancelled booking');
      }

      // Calculate refund amount
      const refundAmount = this.pricingService.calculateRefund(booking);

      // Process refund
      if (refundAmount > 0) {
        await this.paymentService.processRefund(booking.paymentId, refundAmount);
      }

      // Update booking status
      await this.databaseService.updateBookingStatus(bookingId, 'cancelled');

      // Update space status
      await this.databaseService.updateSpaceStatus(booking.spaceId, 'available');

      // Cache invalidation
      await this.redisService.del(`available_spaces:${booking.lotId}:*`);

      // Send notification
      await this.notificationService.sendBookingCancellation(booking.userId, booking, refundAmount);

      // WebSocket notification
      await this.webSocketService.broadcast('booking_cancelled', {
        bookingId,
        spaceId: booking.spaceId,
        refundAmount
      });

      this.logger.info('Booking cancelled successfully', { 
        bookingId, 
        userId, 
        refundAmount 
      });

      return {
        success: true,
        refundAmount
      };
    } catch (error) {
      this.logger.error('Failed to cancel booking', { bookingId, userId, error });
      throw error;
    }
  }

  /**
   * Get user's booking history
   */
  async getUserBookings(userId: string, limit: number = 10, offset: number = 0): Promise<Booking[]> {
    try {
      const bookings = await this.databaseService.getUserBookings(userId, limit, offset);
      this.logger.info('Retrieved user bookings', { userId, count: bookings.length });
      return bookings;
    } catch (error) {
      this.logger.error('Failed to get user bookings', { userId, error });
      throw error;
    }
  }

  /**
   * Get real-time parking status
   */
  async getParkingStatus(lotId: string): Promise<{
    totalSpaces: number;
    occupiedSpaces: number;
    availableSpaces: number;
    occupancyRate: number;
  }> {
    try {
      const status = await this.databaseService.getParkingStatus(lotId);
      
      // Cache for 30 seconds
      await this.redisService.setex(`parking_status:${lotId}`, 30, JSON.stringify(status));

      return status;
    } catch (error) {
      this.logger.error('Failed to get parking status', { lotId, error });
      throw error;
    }
  }

  /**
   * Generate unique booking ID
   */
  private generateBookingId(): string {
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substr(2, 5);
    return `BK${timestamp}${random}`.toUpperCase();
  }
}
