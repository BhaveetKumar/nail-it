import { ParkingSpace, UserType, SpacePreferences, AllocationResult } from '../../models/ParkingModels';
import { DatabaseService } from '../database/DatabaseService';
import { Logger } from '../../utils/Logger';

export class SpaceAllocationService {
  private databaseService: DatabaseService;
  private logger: Logger;

  constructor() {
    this.databaseService = new DatabaseService();
    this.logger = new Logger('SpaceAllocationService');
  }

  /**
   * Get available spaces based on criteria
   */
  async getAvailableSpaces(
    lotId: string,
    userType: UserType,
    startTime: Date,
    endTime: Date,
    preferences?: SpacePreferences
  ): Promise<ParkingSpace[]> {
    try {
      // Get all spaces in the lot
      const allSpaces = await this.databaseService.getSpacesByLotId(lotId);
      
      // Filter available spaces
      const availableSpaces = allSpaces.filter(space => 
        space.status === 'available' || space.status === 'reserved'
      );

      // Apply user type restrictions
      const userTypeSpaces = this.filterByUserType(availableSpaces, userType);

      // Apply time-based availability
      const timeAvailableSpaces = await this.filterByTimeAvailability(
        userTypeSpaces,
        startTime,
        endTime
      );

      // Apply preferences
      const preferredSpaces = preferences 
        ? this.applyPreferences(timeAvailableSpaces, preferences)
        : timeAvailableSpaces;

      this.logger.info('Retrieved available spaces', {
        lotId,
        userType,
        totalSpaces: allSpaces.length,
        availableSpaces: preferredSpaces.length
      });

      return preferredSpaces;
    } catch (error) {
      this.logger.error('Failed to get available spaces', {
        lotId,
        userType,
        error
      });
      throw error;
    }
  }

  /**
   * Allocate the best space for a booking
   */
  async allocateSpace(
    lotId: string,
    userType: UserType,
    startTime: Date,
    endTime: Date,
    preferences?: SpacePreferences
  ): Promise<AllocationResult> {
    try {
      const availableSpaces = await this.getAvailableSpaces(
        lotId,
        userType,
        startTime,
        endTime,
        preferences
      );

      if (availableSpaces.length === 0) {
        throw new Error('No available spaces found');
      }

      // Calculate allocation scores
      const scoredSpaces = await this.calculateAllocationScores(
        availableSpaces,
        userType,
        preferences
      );

      // Sort by score (highest first)
      scoredSpaces.sort((a, b) => b.score - a.score);

      const bestAllocation = scoredSpaces[0];

      this.logger.info('Space allocated', {
        spaceId: bestAllocation.space.id,
        score: bestAllocation.score,
        reasons: bestAllocation.reasons
      });

      return bestAllocation;
    } catch (error) {
      this.logger.error('Failed to allocate space', {
        lotId,
        userType,
        error
      });
      throw error;
    }
  }

  /**
   * Filter spaces by user type restrictions
   */
  private filterByUserType(spaces: ParkingSpace[], userType: UserType): ParkingSpace[] {
    return spaces.filter(space => {
      switch (userType) {
        case UserType.EMPLOYEE:
          return ['regular', 'handicap', 'electric', 'vip'].includes(space.type);
        case UserType.VIP:
          return ['vip', 'regular', 'handicap', 'electric'].includes(space.type);
        case UserType.VISITOR:
          return ['regular', 'handicap'].includes(space.type);
        case UserType.CONTRACTOR:
          return ['regular'].includes(space.type);
        default:
          return false;
      }
    });
  }

  /**
   * Filter spaces by time availability
   */
  private async filterByTimeAvailability(
    spaces: ParkingSpace[],
    startTime: Date,
    endTime: Date
  ): Promise<ParkingSpace[]> {
    const availableSpaces: ParkingSpace[] = [];

    for (const space of spaces) {
      const isAvailable = await this.isSpaceAvailableForTime(
        space.id,
        startTime,
        endTime
      );

      if (isAvailable) {
        availableSpaces.push(space);
      }
    }

    return availableSpaces;
  }

  /**
   * Check if space is available for specific time period
   */
  private async isSpaceAvailableForTime(
    spaceId: string,
    startTime: Date,
    endTime: Date
  ): Promise<boolean> {
    try {
      // Check for overlapping bookings
      const overlappingBookings = await this.databaseService.getOverlappingBookings(
        spaceId,
        startTime,
        endTime
      );

      return overlappingBookings.length === 0;
    } catch (error) {
      this.logger.error('Failed to check space availability', {
        spaceId,
        startTime,
        endTime,
        error
      });
      return false;
    }
  }

  /**
   * Apply user preferences to space selection
   */
  private applyPreferences(
    spaces: ParkingSpace[],
    preferences: SpacePreferences
  ): ParkingSpace[] {
    let filteredSpaces = spaces;

    // Filter by space type
    if (preferences.spaceType && preferences.spaceType.length > 0) {
      filteredSpaces = filteredSpaces.filter(space =>
        preferences.spaceType!.includes(space.type)
      );
    }

    // Filter by floor
    if (preferences.floor && preferences.floor.length > 0) {
      filteredSpaces = filteredSpaces.filter(space =>
        preferences.floor!.includes(space.location.floor)
      );
    }

    // Filter by features
    if (preferences.features && preferences.features.length > 0) {
      filteredSpaces = filteredSpaces.filter(space =>
        preferences.features!.every(feature =>
          space.features.includes(feature)
        )
      );
    }

    // Filter by accessibility
    if (preferences.accessibility) {
      filteredSpaces = filteredSpaces.filter(space =>
        space.type === 'handicap' || space.features.includes('accessible')
      );
    }

    return filteredSpaces;
  }

  /**
   * Calculate allocation scores for spaces
   */
  private async calculateAllocationScores(
    spaces: ParkingSpace[],
    userType: UserType,
    preferences?: SpacePreferences
  ): Promise<AllocationResult[]> {
    const results: AllocationResult[] = [];

    for (const space of spaces) {
      let score = 0;
      const reasons: string[] = [];

      // Base score
      score += 100;
      reasons.push('Available space');

      // User type priority
      score += this.getUserTypeScore(space, userType);
      reasons.push(`${userType} user type match`);

      // Space type preference
      if (preferences?.spaceType?.includes(space.type)) {
        score += 50;
        reasons.push('Preferred space type');
      }

      // Floor preference
      if (preferences?.floor?.includes(space.location.floor)) {
        score += 30;
        reasons.push('Preferred floor');
      }

      // Feature preference
      if (preferences?.features) {
        const matchingFeatures = space.features.filter(feature =>
          preferences.features!.includes(feature)
        );
        score += matchingFeatures.length * 10;
        if (matchingFeatures.length > 0) {
          reasons.push(`Matching features: ${matchingFeatures.join(', ')}`);
        }
      }

      // Accessibility preference
      if (preferences?.accessibility && space.type === 'handicap') {
        score += 40;
        reasons.push('Accessibility requirement met');
      }

      // Distance from entrance (if specified)
      if (preferences?.maxDistance) {
        const distance = this.calculateDistanceFromEntrance(space);
        if (distance <= preferences.maxDistance) {
          score += Math.max(0, 20 - distance);
          reasons.push('Close to entrance');
        }
      }

      // Historical usage (prefer less used spaces)
      const usageScore = await this.getHistoricalUsageScore(space.id);
      score += usageScore;
      if (usageScore > 0) {
        reasons.push('Low historical usage');
      }

      // Rate preference (prefer lower rates for same quality)
      const rateScore = this.getRateScore(space, userType);
      score += rateScore;
      if (rateScore > 0) {
        reasons.push('Good value for money');
      }

      results.push({
        space,
        score: Math.max(0, score),
        reasons
      });
    }

    return results;
  }

  /**
   * Get user type specific score
   */
  private getUserTypeScore(space: ParkingSpace, userType: UserType): number {
    const scores = {
      [UserType.EMPLOYEE]: {
        regular: 20,
        handicap: 15,
        electric: 25,
        vip: 10
      },
      [UserType.VIP]: {
        regular: 10,
        handicap: 5,
        electric: 15,
        vip: 30
      },
      [UserType.VISITOR]: {
        regular: 25,
        handicap: 20,
        electric: 0,
        vip: 0
      },
      [UserType.CONTRACTOR]: {
        regular: 30,
        handicap: 0,
        electric: 0,
        vip: 0
      }
    };

    return scores[userType][space.type as keyof typeof scores[UserType]] || 0;
  }

  /**
   * Calculate distance from entrance
   */
  private calculateDistanceFromEntrance(space: ParkingSpace): number {
    // Simple distance calculation based on floor and position
    // In a real implementation, this would use actual coordinates
    const floorDistance = space.location.floor * 10;
    const positionDistance = Math.abs(space.location.column - 1) * 2;
    return floorDistance + positionDistance;
  }

  /**
   * Get historical usage score
   */
  private async getHistoricalUsageScore(spaceId: string): Promise<number> {
    try {
      const usageCount = await this.databaseService.getSpaceUsageCount(spaceId);
      
      // Lower usage = higher score
      if (usageCount < 10) return 20;
      if (usageCount < 50) return 10;
      if (usageCount < 100) return 5;
      return 0;
    } catch (error) {
      this.logger.error('Failed to get historical usage score', {
        spaceId,
        error
      });
      return 0;
    }
  }

  /**
   * Get rate-based score
   */
  private getRateScore(space: ParkingSpace, userType: UserType): number {
    // Prefer spaces with good value for money
    const baseRate = space.hourlyRate;
    
    if (baseRate < 50) return 15;
    if (baseRate < 100) return 10;
    if (baseRate < 200) return 5;
    return 0;
  }

  /**
   * Reserve space temporarily
   */
  async temporaryReserve(spaceId: string, durationMinutes: number): Promise<void> {
    try {
      await this.databaseService.temporaryReserveSpace(spaceId, durationMinutes);
      
      this.logger.info('Space temporarily reserved', {
        spaceId,
        durationMinutes
      });
    } catch (error) {
      this.logger.error('Failed to temporarily reserve space', {
        spaceId,
        durationMinutes,
        error
      });
      throw error;
    }
  }

  /**
   * Release temporary reservation
   */
  async releaseTemporaryReservation(spaceId: string): Promise<void> {
    try {
      await this.databaseService.releaseTemporaryReservation(spaceId);
      
      this.logger.info('Temporary reservation released', { spaceId });
    } catch (error) {
      this.logger.error('Failed to release temporary reservation', {
        spaceId,
        error
      });
      throw error;
    }
  }
}
