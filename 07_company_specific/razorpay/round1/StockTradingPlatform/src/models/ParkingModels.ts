// Parking Lot Management System Models

export interface ParkingLot {
  id: string;
  name: string;
  address: string;
  totalSpaces: number;
  floors: ParkingFloor[];
  pricing: PricingModel;
  operatingHours: OperatingHours;
  amenities: string[];
  status: 'active' | 'maintenance' | 'closed';
  createdAt: Date;
  updatedAt: Date;
}

export interface ParkingFloor {
  id: string;
  lotId: string;
  floorNumber: number;
  name: string;
  spaces: ParkingSpace[];
  totalSpaces: number;
  availableSpaces: number;
  features: string[];
}

export interface ParkingSpace {
  id: string;
  lotId: string;
  floorId: string;
  spaceNumber: string;
  type: 'regular' | 'handicap' | 'electric' | 'vip';
  status: 'available' | 'occupied' | 'reserved' | 'maintenance';
  currentBooking?: Booking;
  location: {
    row: string;
    column: number;
    floor: number;
  };
  features: string[];
  hourlyRate: number;
  dailyRate: number;
  maxDuration: number; // in hours
}

export interface Booking {
  id: string;
  userId: string;
  spaceId: string;
  lotId: string;
  startTime: Date;
  endTime: Date;
  expectedEndTime?: Date;
  status: 'confirmed' | 'active' | 'completed' | 'cancelled' | 'expired';
  paymentId: string;
  totalAmount: number;
  finalAmount?: number;
  vehicleInfo: VehicleInfo;
  checkInTime?: Date;
  checkOutTime?: Date;
  createdAt: Date;
  updatedAt: Date;
}

export interface User {
  id: string;
  email: string;
  name: string;
  phone: string;
  userType: 'employee' | 'visitor' | 'vip' | 'contractor';
  employeeId?: string;
  department?: string;
  vehicleInfo: VehicleInfo[];
  paymentMethods: PaymentMethod[];
  preferences: UserPreferences;
  status: 'active' | 'suspended' | 'inactive';
  createdAt: Date;
  updatedAt: Date;
}

export interface VehicleInfo {
  id: string;
  userId: string;
  licensePlate: string;
  make: string;
  model: string;
  color: string;
  type: 'car' | 'motorcycle' | 'truck' | 'electric';
  rfidTag?: string;
  isDefault: boolean;
  createdAt: Date;
}

export interface PaymentMethod {
  id: string;
  userId: string;
  type: 'card' | 'upi' | 'wallet' | 'netbanking';
  provider: string;
  lastFourDigits?: string;
  isDefault: boolean;
  status: 'active' | 'inactive';
  createdAt: Date;
}

export interface UserPreferences {
  preferredSpaceType: string[];
  notificationSettings: {
    email: boolean;
    sms: boolean;
    push: boolean;
  };
  autoExtend: boolean;
  maxExtensionHours: number;
  reminderMinutes: number[];
}

export interface PricingModel {
  baseRates: {
    hourly: number;
    daily: number;
    weekly: number;
    monthly: number;
  };
  userTypeMultipliers: {
    employee: number;
    visitor: number;
    vip: number;
    contractor: number;
  };
  timeMultipliers: {
    peak: number;
    offPeak: number;
    weekend: number;
    holiday: number;
  };
  spaceTypeMultipliers: {
    regular: number;
    handicap: number;
    electric: number;
    vip: number;
  };
  demandPricing: {
    enabled: boolean;
    maxMultiplier: number;
    thresholds: DemandThreshold[];
  };
}

export interface DemandThreshold {
  occupancyRate: number;
  multiplier: number;
}

export interface OperatingHours {
  monday: TimeSlot;
  tuesday: TimeSlot;
  wednesday: TimeSlot;
  thursday: TimeSlot;
  friday: TimeSlot;
  saturday: TimeSlot;
  sunday: TimeSlot;
  holidays: HolidaySchedule[];
}

export interface TimeSlot {
  open: string; // HH:MM format
  close: string; // HH:MM format
  is24Hours: boolean;
  isClosed: boolean;
}

export interface HolidaySchedule {
  date: string; // YYYY-MM-DD format
  name: string;
  operatingHours: TimeSlot;
}

export interface Payment {
  id: string;
  bookingId: string;
  userId: string;
  amount: number;
  currency: string;
  method: 'card' | 'upi' | 'wallet' | 'netbanking';
  provider: string;
  transactionId: string;
  status: 'pending' | 'completed' | 'failed' | 'refunded';
  refundAmount?: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface AccessLog {
  id: string;
  bookingId: string;
  vehicleId: string;
  gateId: string;
  action: 'entry' | 'exit';
  timestamp: Date;
  rfidTag: string;
  success: boolean;
  reason?: string;
}

export interface Notification {
  id: string;
  userId: string;
  type: 'booking_confirmation' | 'booking_reminder' | 'booking_cancelled' | 'payment_success' | 'payment_failed' | 'space_available' | 'maintenance_alert';
  title: string;
  message: string;
  data?: any;
  sentAt?: Date;
  status: 'pending' | 'sent' | 'failed';
  channels: ('email' | 'sms' | 'push')[];
}

export interface SpacePreferences {
  spaceType?: string[];
  floor?: number[];
  features?: string[];
  maxDistance?: number; // from entrance
  accessibility?: boolean;
}

export interface AllocationResult {
  space: ParkingSpace;
  score: number;
  reasons: string[];
}

export interface AccessResult {
  allowed: boolean;
  reason?: string;
  bookingId?: string;
}

export interface ParkingStatus {
  totalSpaces: number;
  occupiedSpaces: number;
  availableSpaces: number;
  occupancyRate: number;
  byType: {
    regular: { total: number; occupied: number; available: number };
    handicap: { total: number; occupied: number; available: number };
    electric: { total: number; occupied: number; available: number };
    vip: { total: number; occupied: number; available: number };
  };
  byFloor: {
    [floorNumber: number]: {
      total: number;
      occupied: number;
      available: number;
      occupancyRate: number;
    };
  };
}

export interface BookingSearchCriteria {
  lotId?: string;
  userId?: string;
  status?: string;
  startDate?: Date;
  endDate?: Date;
  spaceType?: string;
  userType?: string;
  limit?: number;
  offset?: number;
}

export interface RevenueReport {
  period: {
    start: Date;
    end: Date;
  };
  totalRevenue: number;
  totalBookings: number;
  averageBookingValue: number;
  byUserType: {
    [userType: string]: {
      revenue: number;
      bookings: number;
      averageValue: number;
    };
  };
  bySpaceType: {
    [spaceType: string]: {
      revenue: number;
      bookings: number;
      averageValue: number;
    };
  };
  byDay: {
    [date: string]: {
      revenue: number;
      bookings: number;
    };
  };
}

export interface MaintenanceRecord {
  id: string;
  spaceId: string;
  lotId: string;
  type: 'cleaning' | 'repair' | 'inspection' | 'upgrade';
  description: string;
  scheduledAt: Date;
  completedAt?: Date;
  assignedTo: string;
  status: 'scheduled' | 'in_progress' | 'completed' | 'cancelled';
  notes?: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface Alert {
  id: string;
  type: 'security' | 'maintenance' | 'system' | 'payment';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  data?: any;
  acknowledged: boolean;
  acknowledgedBy?: string;
  acknowledgedAt?: Date;
  createdAt: Date;
  resolvedAt?: Date;
}

export interface SystemMetrics {
  timestamp: Date;
  totalRequests: number;
  averageResponseTime: number;
  errorRate: number;
  activeConnections: number;
  databaseConnections: number;
  cacheHitRate: number;
  memoryUsage: number;
  cpuUsage: number;
}

// Enums
export enum UserType {
  EMPLOYEE = 'employee',
  VISITOR = 'visitor',
  VIP = 'vip',
  CONTRACTOR = 'contractor'
}

export enum SpaceType {
  REGULAR = 'regular',
  HANDICAP = 'handicap',
  ELECTRIC = 'electric',
  VIP = 'vip'
}

export enum BookingStatus {
  CONFIRMED = 'confirmed',
  ACTIVE = 'active',
  COMPLETED = 'completed',
  CANCELLED = 'cancelled',
  EXPIRED = 'expired'
}

export enum SpaceStatus {
  AVAILABLE = 'available',
  OCCUPIED = 'occupied',
  RESERVED = 'reserved',
  MAINTENANCE = 'maintenance'
}

export enum PaymentStatus {
  PENDING = 'pending',
  COMPLETED = 'completed',
  FAILED = 'failed',
  REFUNDED = 'refunded'
}

export enum NotificationType {
  BOOKING_CONFIRMATION = 'booking_confirmation',
  BOOKING_REMINDER = 'booking_reminder',
  BOOKING_CANCELLED = 'booking_cancelled',
  PAYMENT_SUCCESS = 'payment_success',
  PAYMENT_FAILED = 'payment_failed',
  SPACE_AVAILABLE = 'space_available',
  MAINTENANCE_ALERT = 'maintenance_alert'
}

export enum AlertType {
  SECURITY = 'security',
  MAINTENANCE = 'maintenance',
  SYSTEM = 'system',
  PAYMENT = 'payment'
}

export enum AlertSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}
