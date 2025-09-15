import { ParkingSpace, UserType, Booking, PricingModel } from '../../models/ParkingModels';
import { DatabaseService } from '../database/DatabaseService';
import { Logger } from '../../utils/Logger';

export class PricingService {
  private databaseService: DatabaseService;
  private logger: Logger;

  constructor() {
    this.databaseService = new DatabaseService();
    this.logger = new Logger('PricingService');
  }

  /**
   * Calculate price for a parking space
   */
  calculatePrice(
    space: ParkingSpace,
    startTime: Date,
    endTime: Date,
    userType: UserType
  ): number {
    try {
      const baseRate = space.hourlyRate;
      const duration = this.calculateDuration(startTime, endTime);
      
      // Apply user type discount
      const userDiscount = this.getUserTypeDiscount(userType);
      
      // Apply time-based multipliers
      const timeMultiplier = this.getTimeMultiplier(startTime);
      
      // Apply space type multiplier
      const spaceTypeMultiplier = this.getSpaceTypeMultiplier(space.type);
      
      // Apply demand-based pricing
      const demandMultiplier = this.getDemandMultiplier(space.lotId, startTime);
      
      const finalPrice = baseRate * duration * userDiscount * timeMultiplier * spaceTypeMultiplier * demandMultiplier;
      
      this.logger.info('Price calculated', {
        spaceId: space.id,
        userType,
        duration,
        baseRate,
        finalPrice,
        multipliers: {
          userDiscount,
          timeMultiplier,
          spaceTypeMultiplier,
          demandMultiplier
        }
      });
      
      return Math.round(finalPrice * 100) / 100; // Round to 2 decimal places
    } catch (error) {
      this.logger.error('Failed to calculate price', {
        spaceId: space.id,
        userType,
        startTime,
        endTime,
        error
      });
      throw error;
    }
  }

  /**
   * Calculate final price for a completed booking
   */
  calculateFinalPrice(booking: Booking): number {
    try {
      if (!booking.checkInTime || !booking.checkOutTime) {
        return booking.totalAmount;
      }

      const actualDuration = this.calculateDuration(booking.checkInTime, booking.checkOutTime);
      const expectedDuration = this.calculateDuration(booking.startTime, booking.endTime);
      
      // If actual duration is within 15 minutes of expected, use original price
      if (Math.abs(actualDuration - expectedDuration) <= 0.25) {
        return booking.totalAmount;
      }

      // Recalculate based on actual duration
      const space = this.databaseService.getSpaceById(booking.spaceId);
      if (!space) {
        return booking.totalAmount;
      }

      const user = this.databaseService.getUserById(booking.userId);
      if (!user) {
        return booking.totalAmount;
      }

      return this.calculatePrice(space, booking.checkInTime, booking.checkOutTime, user.userType);
    } catch (error) {
      this.logger.error('Failed to calculate final price', {
        bookingId: booking.id,
        error
      });
      return booking.totalAmount;
    }
  }

  /**
   * Calculate refund amount for cancelled booking
   */
  calculateRefund(booking: Booking): number {
    try {
      const now = new Date();
      const timeUntilStart = booking.startTime.getTime() - now.getTime();
      const hoursUntilStart = timeUntilStart / (1000 * 60 * 60);

      // Full refund if cancelled more than 2 hours before start
      if (hoursUntilStart > 2) {
        return booking.totalAmount;
      }

      // 50% refund if cancelled 1-2 hours before start
      if (hoursUntilStart > 1) {
        return booking.totalAmount * 0.5;
      }

      // 25% refund if cancelled 30 minutes to 1 hour before start
      if (hoursUntilStart > 0.5) {
        return booking.totalAmount * 0.25;
      }

      // No refund if cancelled less than 30 minutes before start
      return 0;
    } catch (error) {
      this.logger.error('Failed to calculate refund', {
        bookingId: booking.id,
        error
      });
      return 0;
    }
  }

  /**
   * Calculate duration in hours
   */
  private calculateDuration(startTime: Date, endTime: Date): number {
    const diffMs = endTime.getTime() - startTime.getTime();
    return diffMs / (1000 * 60 * 60); // Convert to hours
  }

  /**
   * Get user type discount multiplier
   */
  private getUserTypeDiscount(userType: UserType): number {
    const discounts = {
      [UserType.EMPLOYEE]: 0.0, // Free for employees
      [UserType.VIP]: 0.5, // 50% discount for VIP
      [UserType.VISITOR]: 1.0, // Full price for visitors
      [UserType.CONTRACTOR]: 0.8 // 20% discount for contractors
    };

    return discounts[userType] || 1.0;
  }

  /**
   * Get time-based multiplier
   */
  private getTimeMultiplier(startTime: Date): number {
    const hour = startTime.getHours();
    const day = startTime.getDay();

    // Weekend multiplier
    if (day === 0 || day === 6) {
      return 1.2; // 20% higher on weekends
    }

    // Peak hours (8-10 AM, 5-7 PM)
    if ((hour >= 8 && hour < 10) || (hour >= 17 && hour < 19)) {
      return 1.5; // 50% higher during peak hours
    }

    // Off-peak hours (10 PM - 6 AM)
    if (hour >= 22 || hour < 6) {
      return 0.7; // 30% lower during off-peak hours
    }

    // Regular hours
    return 1.0;
  }

  /**
   * Get space type multiplier
   */
  private getSpaceTypeMultiplier(spaceType: string): number {
    const multipliers = {
      'regular': 1.0,
      'handicap': 0.8, // 20% discount for accessibility
      'electric': 1.2, // 20% premium for electric vehicle charging
      'vip': 2.0 // 100% premium for VIP spaces
    };

    return multipliers[spaceType as keyof typeof multipliers] || 1.0;
  }

  /**
   * Get demand-based multiplier
   */
  private getDemandMultiplier(lotId: string, startTime: Date): number {
    try {
      // In a real implementation, this would check current occupancy
      // For now, return a base multiplier
      const occupancyRate = this.getCurrentOccupancyRate(lotId);
      
      if (occupancyRate > 0.9) {
        return 1.5; // 50% higher when 90%+ occupied
      } else if (occupancyRate > 0.8) {
        return 1.3; // 30% higher when 80%+ occupied
      } else if (occupancyRate > 0.7) {
        return 1.1; // 10% higher when 70%+ occupied
      }

      return 1.0; // No demand multiplier
    } catch (error) {
      this.logger.error('Failed to get demand multiplier', {
        lotId,
        startTime,
        error
      });
      return 1.0;
    }
  }

  /**
   * Get current occupancy rate for a lot
   */
  private getCurrentOccupancyRate(lotId: string): number {
    try {
      // In a real implementation, this would query the database
      // For now, return a mock value
      return Math.random() * 0.5 + 0.3; // Random between 30% and 80%
    } catch (error) {
      this.logger.error('Failed to get occupancy rate', {
        lotId,
        error
      });
      return 0.5; // Default 50% occupancy
    }
  }

  /**
   * Get pricing model for a parking lot
   */
  async getPricingModel(lotId: string): Promise<PricingModel | null> {
    try {
      const lot = await this.databaseService.getParkingLotById(lotId);
      return lot?.pricing || null;
    } catch (error) {
      this.logger.error('Failed to get pricing model', {
        lotId,
        error
      });
      return null;
    }
  }

  /**
   * Update pricing model for a parking lot
   */
  async updatePricingModel(lotId: string, pricing: PricingModel): Promise<boolean> {
    try {
      await this.databaseService.updatePricingModel(lotId, pricing);
      
      this.logger.info('Pricing model updated', {
        lotId,
        pricing
      });
      
      return true;
    } catch (error) {
      this.logger.error('Failed to update pricing model', {
        lotId,
        pricing,
        error
      });
      return false;
    }
  }

  /**
   * Calculate bulk pricing for multiple spaces
   */
  calculateBulkPricing(
    spaces: ParkingSpace[],
    startTime: Date,
    endTime: Date,
    userType: UserType
  ): { spaceId: string; price: number }[] {
    try {
      return spaces.map(space => ({
        spaceId: space.id,
        price: this.calculatePrice(space, startTime, endTime, userType)
      }));
    } catch (error) {
      this.logger.error('Failed to calculate bulk pricing', {
        spaceCount: spaces.length,
        userType,
        startTime,
        endTime,
        error
      });
      throw error;
    }
  }

  /**
   * Get price history for a space
   */
  async getPriceHistory(
    spaceId: string,
    startDate: Date,
    endDate: Date
  ): Promise<{ date: Date; price: number; occupancy: number }[]> {
    try {
      const history = await this.databaseService.getPriceHistory(spaceId, startDate, endDate);
      
      this.logger.info('Retrieved price history', {
        spaceId,
        startDate,
        endDate,
        recordCount: history.length
      });
      
      return history;
    } catch (error) {
      this.logger.error('Failed to get price history', {
        spaceId,
        startDate,
        endDate,
        error
      });
      throw error;
    }
  }

  /**
   * Calculate average price for a time period
   */
  async getAveragePrice(
    lotId: string,
    startDate: Date,
    endDate: Date,
    spaceType?: string
  ): Promise<number> {
    try {
      const averagePrice = await this.databaseService.getAveragePrice(
        lotId,
        startDate,
        endDate,
        spaceType
      );
      
      this.logger.info('Retrieved average price', {
        lotId,
        startDate,
        endDate,
        spaceType,
        averagePrice
      });
      
      return averagePrice;
    } catch (error) {
      this.logger.error('Failed to get average price', {
        lotId,
        startDate,
        endDate,
        spaceType,
        error
      });
      throw error;
    }
  }
}
