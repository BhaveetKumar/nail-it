import { Router } from 'express';
import { ParkingController } from './parkingController';

const router = Router();
const parkingController = new ParkingController();

// Health check
router.get('/health', parkingController.healthCheck);

// Parking lots
router.get('/lots', parkingController.getParkingLots);
router.get('/lots/:id', parkingController.getParkingLotById);
router.get('/lots/:id/spaces/available', parkingController.getAvailableSpaces);
router.get('/lots/:id/status', parkingController.getParkingStatus);

// Bookings
router.post('/bookings', parkingController.createBooking);
router.get('/bookings', parkingController.getUserBookings);
router.get('/bookings/:id', parkingController.getBookingById);
router.delete('/bookings/:id', parkingController.cancelBooking);

// Access control
router.post('/entry', parkingController.processEntry);
router.post('/exit', parkingController.processExit);

export default router;
