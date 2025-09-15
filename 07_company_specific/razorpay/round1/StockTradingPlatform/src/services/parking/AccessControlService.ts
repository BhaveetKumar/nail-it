import { AccessResult, VehicleInfo, Booking } from '../../models/ParkingModels';
import { DatabaseService } from '../database/DatabaseService';
import { Logger } from '../../utils/Logger';

export class AccessControlService {
  private databaseService: DatabaseService;
  private logger: Logger;

  constructor() {
    this.databaseService = new DatabaseService();
    this.logger = new Logger('AccessControlService');
  }

  /**
   * Process vehicle entry
   */
  async processEntry(rfidTag: string, gateId: string): Promise<AccessResult> {
    try {
      this.logger.info('Processing entry', { rfidTag, gateId });

      // 1. Validate RFID tag
      const vehicle = await this.getVehicleByRFID(rfidTag);
      if (!vehicle) {
        this.logger.warn('Invalid RFID tag', { rfidTag, gateId });
        return { allowed: false, reason: 'Invalid RFID tag' };
      }

      // 2. Check active booking
      const activeBooking = await this.getActiveBooking(vehicle.userId);
      if (!activeBooking) {
        this.logger.warn('No active booking found', { rfidTag, gateId, userId: vehicle.userId });
        return { allowed: false, reason: 'No active booking found' };
      }

      // 3. Verify space assignment
      const assignedSpace = await this.databaseService.getSpaceById(activeBooking.spaceId);
      if (!assignedSpace) {
        this.logger.warn('Assigned space not found', { rfidTag, gateId, spaceId: activeBooking.spaceId });
        return { allowed: false, reason: 'Assigned space not found' };
      }

      // 4. Check if this is the correct entrance for the space
      if (!this.isCorrectEntrance(gateId, assignedSpace)) {
        this.logger.warn('Wrong entrance gate', { rfidTag, gateId, spaceId: activeBooking.spaceId });
        return { allowed: false, reason: 'Wrong entrance gate for this space' };
      }

      // 5. Check if booking is within time window
      const now = new Date();
      const timeDiff = activeBooking.startTime.getTime() - now.getTime();
      const minutesUntilStart = timeDiff / (1000 * 60);

      // Allow entry 15 minutes before start time
      if (minutesUntilStart > 15) {
        this.logger.warn('Too early for entry', { 
          rfidTag, 
          gateId, 
          startTime: activeBooking.startTime,
          minutesUntilStart 
        });
        return { allowed: false, reason: 'Too early for entry (15 minutes before start time)' };
      }

      // 6. Check if booking has expired
      if (now > activeBooking.endTime) {
        this.logger.warn('Booking expired', { 
          rfidTag, 
          gateId, 
          endTime: activeBooking.endTime 
        });
        return { allowed: false, reason: 'Booking has expired' };
      }

      // 7. Record entry
      await this.recordEntry(activeBooking.id, gateId, rfidTag);

      // 8. Update booking status to active
      await this.databaseService.updateBookingStatus(activeBooking.id, 'active');

      this.logger.info('Entry granted', { 
        rfidTag, 
        gateId, 
        bookingId: activeBooking.id,
        spaceId: activeBooking.spaceId 
      });

      return { 
        allowed: true, 
        bookingId: activeBooking.id 
      };

    } catch (error) {
      this.logger.error('Failed to process entry', { rfidTag, gateId, error });
      return { allowed: false, reason: 'System error during entry processing' };
    }
  }

  /**
   * Process vehicle exit
   */
  async processExit(rfidTag: string, gateId: string): Promise<AccessResult> {
    try {
      this.logger.info('Processing exit', { rfidTag, gateId });

      // 1. Validate RFID tag
      const vehicle = await this.getVehicleByRFID(rfidTag);
      if (!vehicle) {
        this.logger.warn('Invalid RFID tag for exit', { rfidTag, gateId });
        return { allowed: false, reason: 'Invalid RFID tag' };
      }

      // 2. Check active booking
      const activeBooking = await this.getActiveBooking(vehicle.userId);
      if (!activeBooking) {
        this.logger.warn('No active booking found for exit', { rfidTag, gateId, userId: vehicle.userId });
        return { allowed: false, reason: 'No active booking found' };
      }

      // 3. Verify space assignment
      const assignedSpace = await this.databaseService.getSpaceById(activeBooking.spaceId);
      if (!assignedSpace) {
        this.logger.warn('Assigned space not found for exit', { rfidTag, gateId, spaceId: activeBooking.spaceId });
        return { allowed: false, reason: 'Assigned space not found' };
      }

      // 4. Check if this is the correct exit for the space
      if (!this.isCorrectExit(gateId, assignedSpace)) {
        this.logger.warn('Wrong exit gate', { rfidTag, gateId, spaceId: activeBooking.spaceId });
        return { allowed: false, reason: 'Wrong exit gate for this space' };
      }

      // 5. Record exit
      await this.recordExit(activeBooking.id, gateId, rfidTag);

      // 6. Update booking status to completed
      await this.databaseService.updateBookingStatus(activeBooking.id, 'completed');

      // 7. Update space status to available
      await this.databaseService.updateSpaceStatus(activeBooking.spaceId, 'available');

      this.logger.info('Exit granted', { 
        rfidTag, 
        gateId, 
        bookingId: activeBooking.id,
        spaceId: activeBooking.spaceId 
      });

      return { 
        allowed: true, 
        bookingId: activeBooking.id 
      };

    } catch (error) {
      this.logger.error('Failed to process exit', { rfidTag, gateId, error });
      return { allowed: false, reason: 'System error during exit processing' };
    }
  }

  /**
   * Get vehicle by RFID tag
   */
  private async getVehicleByRFID(rfidTag: string): Promise<VehicleInfo | null> {
    try {
      return await this.databaseService.getVehicleByRFID(rfidTag);
    } catch (error) {
      this.logger.error('Failed to get vehicle by RFID', { rfidTag, error });
      return null;
    }
  }

  /**
   * Get active booking for user
   */
  private async getActiveBooking(userId: string): Promise<Booking | null> {
    try {
      return await this.databaseService.getActiveBooking(userId);
    } catch (error) {
      this.logger.error('Failed to get active booking', { userId, error });
      return null;
    }
  }

  /**
   * Check if gate is correct entrance for space
   */
  private isCorrectEntrance(gateId: string, space: any): boolean {
    // In a real implementation, this would check gate-to-space mapping
    // For now, we'll assume all gates can access all spaces
    return true;
  }

  /**
   * Check if gate is correct exit for space
   */
  private isCorrectExit(gateId: string, space: any): boolean {
    // In a real implementation, this would check gate-to-space mapping
    // For now, we'll assume all gates can access all spaces
    return true;
  }

  /**
   * Record entry in access log
   */
  private async recordEntry(bookingId: string, gateId: string, rfidTag: string): Promise<void> {
    try {
      const accessLog = {
        id: this.generateAccessLogId(),
        bookingId,
        vehicleId: '', // Will be filled by database service
        gateId,
        action: 'entry',
        timestamp: new Date(),
        rfidTag,
        success: true
      };

      await this.databaseService.createAccessLog(accessLog);
      
      this.logger.info('Entry recorded', { bookingId, gateId, rfidTag });
    } catch (error) {
      this.logger.error('Failed to record entry', { bookingId, gateId, rfidTag, error });
      throw error;
    }
  }

  /**
   * Record exit in access log
   */
  private async recordExit(bookingId: string, gateId: string, rfidTag: string): Promise<void> {
    try {
      const accessLog = {
        id: this.generateAccessLogId(),
        bookingId,
        vehicleId: '', // Will be filled by database service
        gateId,
        action: 'exit',
        timestamp: new Date(),
        rfidTag,
        success: true
      };

      await this.databaseService.createAccessLog(accessLog);
      
      this.logger.info('Exit recorded', { bookingId, gateId, rfidTag });
    } catch (error) {
      this.logger.error('Failed to record exit', { bookingId, gateId, rfidTag, error });
      throw error;
    }
  }

  /**
   * Generate unique access log ID
   */
  private generateAccessLogId(): string {
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substr(2, 5);
    return `AL${timestamp}${random}`.toUpperCase();
  }

  /**
   * Get access logs for a booking
   */
  async getAccessLogs(bookingId: string): Promise<any[]> {
    try {
      return await this.databaseService.getAccessLogsByBooking(bookingId);
    } catch (error) {
      this.logger.error('Failed to get access logs', { bookingId, error });
      throw error;
    }
  }

  /**
   * Get access logs for a time period
   */
  async getAccessLogsByTimeRange(
    startTime: Date, 
    endTime: Date, 
    gateId?: string
  ): Promise<any[]> {
    try {
      return await this.databaseService.getAccessLogsByTimeRange(startTime, endTime, gateId);
    } catch (error) {
      this.logger.error('Failed to get access logs by time range', { 
        startTime, 
        endTime, 
        gateId, 
        error 
      });
      throw error;
    }
  }

  /**
   * Check if vehicle is currently in parking lot
   */
  async isVehicleInLot(rfidTag: string): Promise<boolean> {
    try {
      const vehicle = await this.getVehicleByRFID(rfidTag);
      if (!vehicle) return false;

      const activeBooking = await this.getActiveBooking(vehicle.userId);
      return activeBooking !== null;
    } catch (error) {
      this.logger.error('Failed to check if vehicle is in lot', { rfidTag, error });
      return false;
    }
  }

  /**
   * Get current occupancy by gate
   */
  async getGateOccupancy(gateId: string): Promise<{
    totalEntries: number;
    totalExits: number;
    currentOccupancy: number;
  }> {
    try {
      const today = new Date();
      const startOfDay = new Date(today.getFullYear(), today.getMonth(), today.getDate());
      const endOfDay = new Date(today.getFullYear(), today.getMonth(), today.getDate() + 1);

      const entries = await this.databaseService.getAccessLogsByTimeRange(
        startOfDay, 
        endOfDay, 
        gateId
      );

      const totalEntries = entries.filter(log => log.action === 'entry').length;
      const totalExits = entries.filter(log => log.action === 'exit').length;
      const currentOccupancy = totalEntries - totalExits;

      return {
        totalEntries,
        totalExits,
        currentOccupancy: Math.max(0, currentOccupancy)
      };
    } catch (error) {
      this.logger.error('Failed to get gate occupancy', { gateId, error });
      throw error;
    }
  }
}
