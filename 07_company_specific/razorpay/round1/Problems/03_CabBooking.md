# 03. Cab Booking - Ride-Hailing System

## Title & Summary
Design and implement a cab booking system that handles ride requests, driver matching, real-time tracking, and payment processing with geolocation services.

## Problem Statement

Build a ride-hailing system that:

1. **User Management**: Register users and drivers with location tracking
2. **Ride Booking**: Allow users to request rides with pickup/drop locations
3. **Driver Matching**: Find and assign nearby available drivers
4. **Real-time Tracking**: Track ride progress and driver location
5. **Payment Processing**: Handle ride payments and driver earnings
6. **Rating System**: Allow users to rate drivers and vice versa

## Requirements & Constraints

### Functional Requirements
- User and driver registration with location
- Ride request with pickup/drop coordinates
- Driver matching based on proximity and availability
- Real-time location tracking during rides
- Payment processing and fare calculation
- Rating and review system

### Non-Functional Requirements
- **Latency**: < 500ms for driver matching
- **Consistency**: Strong consistency for ride state
- **Memory**: Support 10K concurrent users
- **Scalability**: Handle 100K rides per day
- **Reliability**: 99.9% ride completion rate

## API / Interfaces

### REST Endpoints

```go
// User Management
POST   /api/users/register
POST   /api/drivers/register
PUT    /api/users/{userID}/location
PUT    /api/drivers/{driverID}/location
PUT    /api/drivers/{driverID}/status

// Ride Management
POST   /api/rides/request
GET    /api/rides/{rideID}
PUT    /api/rides/{rideID}/accept
PUT    /api/rides/{rideID}/start
PUT    /api/rides/{rideID}/complete
PUT    /api/rides/{rideID}/cancel

// Payment
POST   /api/payments/process
GET    /api/payments/{paymentID}

// Rating
POST   /api/ratings
GET    /api/ratings/{userID}

// WebSocket
WS     /ws/rides/{rideID}
WS     /ws/drivers/{driverID}
```

### Request/Response Examples

```json
// Request Ride
POST /api/rides/request
{
  "userID": "user123",
  "pickupLocation": {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "address": "123 Main St, San Francisco"
  },
  "dropLocation": {
    "latitude": 37.7849,
    "longitude": -122.4094,
    "address": "456 Oak St, San Francisco"
  },
  "rideType": "standard"
}

// Driver Response
{
  "rideID": "ride456",
  "driverID": "driver789",
  "driverName": "John Doe",
  "driverPhone": "+1234567890",
  "vehicleInfo": {
    "make": "Toyota",
    "model": "Camry",
    "licensePlate": "ABC123",
    "color": "White"
  },
  "estimatedArrival": "5 minutes",
  "fare": 15.50
}
```

## Data Model

### Core Entities

```go
type User struct {
    ID       string    `json:"id"`
    Name     string    `json:"name"`
    Email    string    `json:"email"`
    Phone    string    `json:"phone"`
    Location Location  `json:"location"`
    Rating   float64   `json:"rating"`
    CreatedAt time.Time `json:"createdAt"`
}

type Driver struct {
    ID          string    `json:"id"`
    Name        string    `json:"name"`
    Email       string    `json:"email"`
    Phone       string    `json:"phone"`
    Location    Location  `json:"location"`
    Status      DriverStatus `json:"status"`
    Vehicle     Vehicle   `json:"vehicle"`
    Rating      float64   `json:"rating"`
    Earnings    float64   `json:"earnings"`
    CreatedAt   time.Time `json:"createdAt"`
}

type Location struct {
    Latitude  float64 `json:"latitude"`
    Longitude float64 `json:"longitude"`
    Address   string  `json:"address"`
    Timestamp time.Time `json:"timestamp"`
}

type Vehicle struct {
    Make         string `json:"make"`
    Model        string `json:"model"`
    Year         int    `json:"year"`
    LicensePlate string `json:"licensePlate"`
    Color        string `json:"color"`
    Capacity     int    `json:"capacity"`
}

type Ride struct {
    ID            string      `json:"id"`
    UserID        string      `json:"userID"`
    DriverID      *string     `json:"driverID,omitempty"`
    Status        RideStatus  `json:"status"`
    PickupLocation Location   `json:"pickupLocation"`
    DropLocation  Location    `json:"dropLocation"`
    RequestedAt   time.Time   `json:"requestedAt"`
    AcceptedAt    *time.Time  `json:"acceptedAt,omitempty"`
    StartedAt     *time.Time  `json:"startedAt,omitempty"`
    CompletedAt   *time.Time  `json:"completedAt,omitempty"`
    Fare          float64     `json:"fare"`
    Distance      float64     `json:"distance"`
    Duration      int         `json:"duration"` // in minutes
    PaymentID     *string     `json:"paymentID,omitempty"`
}

type Payment struct {
    ID        string        `json:"id"`
    RideID    string        `json:"rideID"`
    Amount    float64       `json:"amount"`
    Status    PaymentStatus `json:"status"`
    Method    string        `json:"method"`
    ProcessedAt time.Time   `json:"processedAt"`
}

type Rating struct {
    ID       string    `json:"id"`
    RideID   string    `json:"rideID"`
    UserID   string    `json:"userID"`
    DriverID string    `json:"driverID"`
    Rating   int       `json:"rating"` // 1-5
    Comment  string    `json:"comment"`
    CreatedAt time.Time `json:"createdAt"`
}
```

## Approach Overview

### Simple Solution (MVP)
1. In-memory storage with maps and slices
2. Simple distance calculation for driver matching
3. Basic ride state management
4. No real-time tracking or payment processing

### Production-Ready Design
1. **Microservices Architecture**: Separate services for users, drivers, rides, payments
2. **Geospatial Indexing**: Use spatial data structures for efficient location queries
3. **Event-Driven**: Use message queues for ride state changes
4. **Real-time Updates**: WebSocket connections for live tracking
5. **Payment Integration**: External payment gateway integration
6. **Caching**: Redis for driver locations and ride states

## Detailed Design

### Modular Decomposition

```go
cabbooking/
├── users/         # User management
├── drivers/       # Driver management
├── rides/         # Ride booking and management
├── payments/      # Payment processing
├── ratings/       # Rating system
├── geolocation/   # Location services
├── websocket/     # Real-time updates
└── workers/       # Background job processing
```

### Concurrency Model

```go
type RideService struct {
    users       map[string]*User
    drivers     map[string]*Driver
    rides       map[string]*Ride
    payments    map[string]*Payment
    ratings     map[string][]Rating
    driverIndex *DriverIndex // Spatial index for drivers
    mutex       sync.RWMutex
    rideChan    chan RideUpdate
    paymentChan chan PaymentRequest
}

// Goroutines for:
// 1. Driver location updates
// 2. Ride state processing
// 3. Payment processing
// 4. WebSocket broadcasting
```

### Persistence Strategy

```go
// Database schema
CREATE TABLE rides (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    driver_id VARCHAR(36),
    status VARCHAR(20) NOT NULL,
    pickup_lat DECIMAL(10,8),
    pickup_lng DECIMAL(11,8),
    drop_lat DECIMAL(10,8),
    drop_lng DECIMAL(11,8),
    fare DECIMAL(10,2),
    distance DECIMAL(10,2),
    requested_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE driver_locations (
    driver_id VARCHAR(36) PRIMARY KEY,
    latitude DECIMAL(10,8),
    longitude DECIMAL(11,8),
    updated_at TIMESTAMP
);
```

## Optimal Golang Implementation

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "math"
    "net/http"
    "sync"
    "time"

    "github.com/gorilla/websocket"
)

type DriverStatus string
const (
    StatusOffline  DriverStatus = "offline"
    StatusOnline   DriverStatus = "online"
    StatusBusy     DriverStatus = "busy"
    StatusInactive DriverStatus = "inactive"
)

type RideStatus string
const (
    RideRequested RideStatus = "requested"
    RideAccepted  RideStatus = "accepted"
    RideStarted   RideStatus = "started"
    RideCompleted RideStatus = "completed"
    RideCancelled RideStatus = "cancelled"
)

type PaymentStatus string
const (
    PaymentPending PaymentStatus = "pending"
    PaymentSuccess PaymentStatus = "success"
    PaymentFailed  PaymentStatus = "failed"
)

type User struct {
    ID       string    `json:"id"`
    Name     string    `json:"name"`
    Email    string    `json:"email"`
    Phone    string    `json:"phone"`
    Location Location  `json:"location"`
    Rating   float64   `json:"rating"`
    CreatedAt time.Time `json:"createdAt"`
}

type Driver struct {
    ID          string    `json:"id"`
    Name        string    `json:"name"`
    Email       string    `json:"email"`
    Phone       string    `json:"phone"`
    Location    Location  `json:"location"`
    Status      DriverStatus `json:"status"`
    Vehicle     Vehicle   `json:"vehicle"`
    Rating      float64   `json:"rating"`
    Earnings    float64   `json:"earnings"`
    CreatedAt   time.Time `json:"createdAt"`
}

type Location struct {
    Latitude  float64 `json:"latitude"`
    Longitude float64 `json:"longitude"`
    Address   string  `json:"address"`
    Timestamp time.Time `json:"timestamp"`
}

type Vehicle struct {
    Make         string `json:"make"`
    Model        string `json:"model"`
    Year         int    `json:"year"`
    LicensePlate string `json:"licensePlate"`
    Color        string `json:"color"`
    Capacity     int    `json:"capacity"`
}

type Ride struct {
    ID            string      `json:"id"`
    UserID        string      `json:"userID"`
    DriverID      *string     `json:"driverID,omitempty"`
    Status        RideStatus  `json:"status"`
    PickupLocation Location   `json:"pickupLocation"`
    DropLocation  Location    `json:"dropLocation"`
    RequestedAt   time.Time   `json:"requestedAt"`
    AcceptedAt    *time.Time  `json:"acceptedAt,omitempty"`
    StartedAt     *time.Time  `json:"startedAt,omitempty"`
    CompletedAt   *time.Time  `json:"completedAt,omitempty"`
    Fare          float64     `json:"fare"`
    Distance      float64     `json:"distance"`
    Duration      int         `json:"duration"`
    PaymentID     *string     `json:"paymentID,omitempty"`
}

type Payment struct {
    ID        string        `json:"id"`
    RideID    string        `json:"rideID"`
    Amount    float64       `json:"amount"`
    Status    PaymentStatus `json:"status"`
    Method    string        `json:"method"`
    ProcessedAt time.Time   `json:"processedAt"`
}

type Rating struct {
    ID       string    `json:"id"`
    RideID   string    `json:"rideID"`
    UserID   string    `json:"userID"`
    DriverID string    `json:"driverID"`
    Rating   int       `json:"rating"`
    Comment  string    `json:"comment"`
    CreatedAt time.Time `json:"createdAt"`
}

type DriverIndex struct {
    drivers []*Driver
    mutex   sync.RWMutex
}

type RideService struct {
    users       map[string]*User
    drivers     map[string]*Driver
    rides       map[string]*Ride
    payments    map[string]*Payment
    ratings     map[string][]Rating
    driverIndex *DriverIndex
    mutex       sync.RWMutex
    rideChan    chan RideUpdate
    paymentChan chan PaymentRequest
    upgrader    websocket.Upgrader
}

type RideUpdate struct {
    RideID string
    Status RideStatus
    Data   interface{}
}

type PaymentRequest struct {
    RideID string
    Amount float64
    Method string
}

func NewRideService() *RideService {
    return &RideService{
        users:       make(map[string]*User),
        drivers:     make(map[string]*Driver),
        rides:       make(map[string]*Ride),
        payments:    make(map[string]*Payment),
        ratings:     make(map[string][]Rating),
        driverIndex: &DriverIndex{drivers: make([]*Driver, 0)},
        rideChan:    make(chan RideUpdate, 1000),
        paymentChan: make(chan PaymentRequest, 1000),
        upgrader: websocket.Upgrader{
            CheckOrigin: func(r *http.Request) bool {
                return true
            },
        },
    }
}

func (rs *RideService) RegisterUser(name, email, phone string) (*User, error) {
    rs.mutex.Lock()
    defer rs.mutex.Unlock()

    // Check if user exists
    for _, user := range rs.users {
        if user.Email == email {
            return nil, fmt.Errorf("user already exists")
        }
    }

    user := &User{
        ID:        fmt.Sprintf("user_%d", time.Now().UnixNano()),
        Name:      name,
        Email:     email,
        Phone:     phone,
        Rating:    5.0,
        CreatedAt: time.Now(),
    }

    rs.users[user.ID] = user
    return user, nil
}

func (rs *RideService) RegisterDriver(name, email, phone string, vehicle Vehicle) (*Driver, error) {
    rs.mutex.Lock()
    defer rs.mutex.Unlock()

    // Check if driver exists
    for _, driver := range rs.drivers {
        if driver.Email == email {
            return nil, fmt.Errorf("driver already exists")
        }
    }

    driver := &Driver{
        ID:        fmt.Sprintf("driver_%d", time.Now().UnixNano()),
        Name:      name,
        Email:     email,
        Phone:     phone,
        Status:    StatusOffline,
        Vehicle:   vehicle,
        Rating:    5.0,
        Earnings:  0.0,
        CreatedAt: time.Now(),
    }

    rs.drivers[driver.ID] = driver
    rs.driverIndex.AddDriver(driver)
    return driver, nil
}

func (rs *RideService) UpdateDriverLocation(driverID string, location Location) error {
    rs.mutex.Lock()
    defer rs.mutex.Unlock()

    driver, exists := rs.drivers[driverID]
    if !exists {
        return fmt.Errorf("driver not found")
    }

    driver.Location = location
    driver.Location.Timestamp = time.Now()
    
    rs.driverIndex.UpdateDriver(driver)
    return nil
}

func (rs *RideService) UpdateDriverStatus(driverID string, status DriverStatus) error {
    rs.mutex.Lock()
    defer rs.mutex.Unlock()

    driver, exists := rs.drivers[driverID]
    if !exists {
        return fmt.Errorf("driver not found")
    }

    driver.Status = status
    rs.driverIndex.UpdateDriver(driver)
    return nil
}

func (rs *RideService) RequestRide(userID string, pickupLocation, dropLocation Location) (*Ride, error) {
    rs.mutex.Lock()
    defer rs.mutex.Unlock()

    user, exists := rs.users[userID]
    if !exists {
        return nil, fmt.Errorf("user not found")
    }

    // Find nearest available driver
    driver := rs.driverIndex.FindNearestDriver(pickupLocation)
    if driver == nil {
        return nil, fmt.Errorf("no drivers available")
    }

    // Calculate fare and distance
    distance := calculateDistance(pickupLocation, dropLocation)
    fare := calculateFare(distance)

    ride := &Ride{
        ID:            fmt.Sprintf("ride_%d", time.Now().UnixNano()),
        UserID:        userID,
        DriverID:      &driver.ID,
        Status:        RideRequested,
        PickupLocation: pickupLocation,
        DropLocation:  dropLocation,
        RequestedAt:   time.Now(),
        Fare:          fare,
        Distance:      distance,
        Duration:      int(distance * 2), // Rough estimate: 2 minutes per km
    }

    rs.rides[ride.ID] = ride

    // Update driver status
    driver.Status = StatusBusy
    rs.driverIndex.UpdateDriver(driver)

    // Send ride update
    rs.rideChan <- RideUpdate{
        RideID: ride.ID,
        Status: RideRequested,
        Data:   ride,
    }

    return ride, nil
}

func (rs *RideService) AcceptRide(rideID, driverID string) error {
    rs.mutex.Lock()
    defer rs.mutex.Unlock()

    ride, exists := rs.rides[rideID]
    if !exists {
        return fmt.Errorf("ride not found")
    }

    if ride.Status != RideRequested {
        return fmt.Errorf("ride cannot be accepted")
    }

    if *ride.DriverID != driverID {
        return fmt.Errorf("driver not assigned to this ride")
    }

    now := time.Now()
    ride.Status = RideAccepted
    ride.AcceptedAt = &now

    rs.rideChan <- RideUpdate{
        RideID: rideID,
        Status: RideAccepted,
        Data:   ride,
    }

    return nil
}

func (rs *RideService) StartRide(rideID string) error {
    rs.mutex.Lock()
    defer rs.mutex.Unlock()

    ride, exists := rs.rides[rideID]
    if !exists {
        return fmt.Errorf("ride not found")
    }

    if ride.Status != RideAccepted {
        return fmt.Errorf("ride cannot be started")
    }

    now := time.Now()
    ride.Status = RideStarted
    ride.StartedAt = &now

    rs.rideChan <- RideUpdate{
        RideID: rideID,
        Status: RideStarted,
        Data:   ride,
    }

    return nil
}

func (rs *RideService) CompleteRide(rideID string) error {
    rs.mutex.Lock()
    defer rs.mutex.Unlock()

    ride, exists := rs.rides[rideID]
    if !exists {
        return fmt.Errorf("ride not found")
    }

    if ride.Status != RideStarted {
        return fmt.Errorf("ride cannot be completed")
    }

    now := time.Now()
    ride.Status = RideCompleted
    ride.CompletedAt = &now

    // Update driver status and earnings
    if ride.DriverID != nil {
        driver := rs.drivers[*ride.DriverID]
        driver.Status = StatusOnline
        driver.Earnings += ride.Fare
        rs.driverIndex.UpdateDriver(driver)
    }

    // Process payment
    payment := &Payment{
        ID:         fmt.Sprintf("payment_%d", time.Now().UnixNano()),
        RideID:     rideID,
        Amount:     ride.Fare,
        Status:     PaymentSuccess,
        Method:     "card",
        ProcessedAt: now,
    }

    rs.payments[payment.ID] = payment
    ride.PaymentID = &payment.ID

    rs.rideChan <- RideUpdate{
        RideID: rideID,
        Status: RideCompleted,
        Data:   ride,
    }

    return nil
}

func (rs *RideService) GetRide(rideID string) (*Ride, error) {
    rs.mutex.RLock()
    defer rs.mutex.RUnlock()

    ride, exists := rs.rides[rideID]
    if !exists {
        return nil, fmt.Errorf("ride not found")
    }

    return ride, nil
}

func (rs *RideService) AddRating(rideID, userID, driverID string, rating int, comment string) error {
    rs.mutex.Lock()
    defer rs.mutex.Unlock()

    ratingObj := Rating{
        ID:       fmt.Sprintf("rating_%d", time.Now().UnixNano()),
        RideID:   rideID,
        UserID:   userID,
        DriverID: driverID,
        Rating:   rating,
        Comment:  comment,
        CreatedAt: time.Now(),
    }

    rs.ratings[driverID] = append(rs.ratings[driverID], ratingObj)

    // Update driver rating
    driver := rs.drivers[driverID]
    if driver != nil {
        driver.Rating = rs.calculateAverageRating(driverID)
    }

    return nil
}

func (rs *RideService) calculateAverageRating(driverID string) float64 {
    ratings := rs.ratings[driverID]
    if len(ratings) == 0 {
        return 5.0
    }

    sum := 0
    for _, rating := range ratings {
        sum += rating.Rating
    }

    return float64(sum) / float64(len(ratings))
}

// DriverIndex for spatial queries
func (di *DriverIndex) AddDriver(driver *Driver) {
    di.mutex.Lock()
    defer di.mutex.Unlock()
    di.drivers = append(di.drivers, driver)
}

func (di *DriverIndex) UpdateDriver(driver *Driver) {
    di.mutex.Lock()
    defer di.mutex.Unlock()
    for i, d := range di.drivers {
        if d.ID == driver.ID {
            di.drivers[i] = driver
            break
        }
    }
}

func (di *DriverIndex) FindNearestDriver(location Location) *Driver {
    di.mutex.RLock()
    defer di.mutex.RUnlock()

    var nearestDriver *Driver
    minDistance := math.MaxFloat64

    for _, driver := range di.drivers {
        if driver.Status != StatusOnline {
            continue
        }

        distance := calculateDistance(location, driver.Location)
        if distance < minDistance {
            minDistance = distance
            nearestDriver = driver
        }
    }

    return nearestDriver
}

// Utility functions
func calculateDistance(loc1, loc2 Location) float64 {
    const earthRadius = 6371 // km
    lat1 := loc1.Latitude * math.Pi / 180
    lat2 := loc2.Latitude * math.Pi / 180
    deltaLat := (loc2.Latitude - loc1.Latitude) * math.Pi / 180
    deltaLng := (loc2.Longitude - loc1.Longitude) * math.Pi / 180

    a := math.Sin(deltaLat/2)*math.Sin(deltaLat/2) +
        math.Cos(lat1)*math.Cos(lat2)*
        math.Sin(deltaLng/2)*math.Sin(deltaLng/2)
    c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))

    return earthRadius * c
}

func calculateFare(distance float64) float64 {
    baseFare := 2.0
    perKmRate := 1.5
    return baseFare + (distance * perKmRate)
}

// HTTP Handlers
func (rs *RideService) RegisterUserHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    var req struct {
        Name  string `json:"name"`
        Email string `json:"email"`
        Phone string `json:"phone"`
    }

    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid request", http.StatusBadRequest)
        return
    }

    user, err := rs.RegisterUser(req.Name, req.Email, req.Phone)
    if err != nil {
        http.Error(w, err.Error(), http.StatusConflict)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

func (rs *RideService) RequestRideHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    var req struct {
        UserID        string   `json:"userID"`
        PickupLocation Location `json:"pickupLocation"`
        DropLocation  Location `json:"dropLocation"`
    }

    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid request", http.StatusBadRequest)
        return
    }

    ride, err := rs.RequestRide(req.UserID, req.PickupLocation, req.DropLocation)
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(ride)
}

func (rs *RideService) GetRideHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodGet {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    rideID := r.URL.Path[len("/api/rides/"):]
    ride, err := rs.GetRide(rideID)
    if err != nil {
        http.Error(w, err.Error(), http.StatusNotFound)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(ride)
}

func main() {
    service := NewRideService()

    // HTTP routes
    http.HandleFunc("/api/users/register", service.RegisterUserHandler)
    http.HandleFunc("/api/rides/request", service.RequestRideHandler)
    http.HandleFunc("/api/rides/", service.GetRideHandler)

    log.Println("Cab booking service starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

## Unit Tests

```go
func TestRideService_RegisterUser(t *testing.T) {
    service := NewRideService()

    user, err := service.RegisterUser("John Doe", "john@example.com", "+1234567890")
    if err != nil {
        t.Fatalf("RegisterUser() error = %v", err)
    }

    if user.Name != "John Doe" {
        t.Errorf("RegisterUser() name = %v, want John Doe", user.Name)
    }

    if user.Email != "john@example.com" {
        t.Errorf("RegisterUser() email = %v, want john@example.com", user.Email)
    }
}

func TestRideService_RequestRide(t *testing.T) {
    service := NewRideService()

    // Register user
    user, _ := service.RegisterUser("John Doe", "john@example.com", "+1234567890")

    // Register driver
    vehicle := Vehicle{Make: "Toyota", Model: "Camry", LicensePlate: "ABC123"}
    driver, _ := service.RegisterDriver("Jane Smith", "jane@example.com", "+0987654321", vehicle)

    // Set driver online
    service.UpdateDriverStatus(driver.ID, StatusOnline)
    service.UpdateDriverLocation(driver.ID, Location{
        Latitude: 37.7749, Longitude: -122.4194,
    })

    // Request ride
    pickupLocation := Location{Latitude: 37.7749, Longitude: -122.4194}
    dropLocation := Location{Latitude: 37.7849, Longitude: -122.4094}

    ride, err := service.RequestRide(user.ID, pickupLocation, dropLocation)
    if err != nil {
        t.Fatalf("RequestRide() error = %v", err)
    }

    if ride.UserID != user.ID {
        t.Errorf("RequestRide() userID = %v, want %v", ride.UserID, user.ID)
    }

    if ride.Status != RideRequested {
        t.Errorf("RequestRide() status = %v, want %v", ride.Status, RideRequested)
    }
}

func TestRideService_CompleteRide(t *testing.T) {
    service := NewRideService()

    // Register user and driver
    user, _ := service.RegisterUser("John Doe", "john@example.com", "+1234567890")
    vehicle := Vehicle{Make: "Toyota", Model: "Camry", LicensePlate: "ABC123"}
    driver, _ := service.RegisterDriver("Jane Smith", "jane@example.com", "+0987654321", vehicle)

    // Set driver online and request ride
    service.UpdateDriverStatus(driver.ID, StatusOnline)
    service.UpdateDriverLocation(driver.ID, Location{Latitude: 37.7749, Longitude: -122.4194})

    pickupLocation := Location{Latitude: 37.7749, Longitude: -122.4194}
    dropLocation := Location{Latitude: 37.7849, Longitude: -122.4094}
    ride, _ := service.RequestRide(user.ID, pickupLocation, dropLocation)

    // Accept and start ride
    service.AcceptRide(ride.ID, driver.ID)
    service.StartRide(ride.ID)

    // Complete ride
    err := service.CompleteRide(ride.ID)
    if err != nil {
        t.Fatalf("CompleteRide() error = %v", err)
    }

    // Check ride status
    updatedRide, _ := service.GetRide(ride.ID)
    if updatedRide.Status != RideCompleted {
        t.Errorf("CompleteRide() status = %v, want %v", updatedRide.Status, RideCompleted)
    }

    // Check driver earnings
    updatedDriver := service.drivers[driver.ID]
    if updatedDriver.Earnings != ride.Fare {
        t.Errorf("CompleteRide() driver earnings = %v, want %v", updatedDriver.Earnings, ride.Fare)
    }
}
```

## Complexity Analysis

### Time Complexity
- **Register User/Driver**: O(1) - Hash map insertion
- **Request Ride**: O(D) - Linear scan through drivers
- **Find Nearest Driver**: O(D) - Linear scan through drivers
- **Complete Ride**: O(1) - Hash map update

### Space Complexity
- **User Storage**: O(U) where U is number of users
- **Driver Storage**: O(D) where D is number of drivers
- **Ride Storage**: O(R) where R is number of rides
- **Total**: O(U + D + R)

## Edge Cases & Validation

### Input Validation
- Invalid coordinates (out of range)
- Empty user/driver information
- Invalid ride states
- Negative fare amounts
- Invalid rating values (1-5)

### Error Scenarios
- No available drivers
- Driver goes offline during ride
- Payment processing failures
- Network connectivity issues
- Invalid ride transitions

### Boundary Conditions
- Maximum ride distance (100 km)
- Minimum fare amount ($1.00)
- Maximum rating value (5)
- Ride timeout (30 minutes)
- Driver response timeout (2 minutes)

## Extension Ideas (Scaling)

### Horizontal Scaling
1. **Load Balancing**: Multiple service instances
2. **Database Sharding**: Partition by geographic regions
3. **Message Queue**: Kafka for ride state changes
4. **Cache Clustering**: Redis cluster for driver locations

### Performance Optimization
1. **Geospatial Indexing**: Use R-tree or quadtree for driver queries
2. **Driver Pooling**: Pre-allocate drivers for popular areas
3. **Predictive Matching**: ML-based driver assignment
4. **Route Optimization**: Real-time traffic data integration

### Advanced Features
1. **Surge Pricing**: Dynamic pricing based on demand
2. **Ride Sharing**: Multiple passengers in same vehicle
3. **Scheduled Rides**: Book rides in advance
4. **Multi-modal Transport**: Integration with public transport

## 20 Follow-up Questions

### 1. How would you handle driver availability in real-time?
**Answer**: Use WebSocket connections for driver status updates. Implement heartbeat mechanism to detect offline drivers. Use Redis for driver location caching with TTL. Consider using geospatial data structures for efficient proximity queries.

### 2. What happens if a driver cancels after accepting a ride?
**Answer**: Implement driver cancellation with penalty system. Use fallback driver matching for cancelled rides. Implement driver reputation scoring. Consider using backup driver pools for critical areas.

### 3. How do you ensure fair driver assignment?
**Answer**: Implement round-robin assignment with driver preferences. Use driver earnings balancing algorithms. Consider driver rating and performance metrics. Implement driver choice system for ride acceptance.

### 4. What's your strategy for handling peak demand?
**Answer**: Implement surge pricing with demand-based multipliers. Use driver incentives for peak hours. Implement ride sharing for high-demand areas. Consider using predictive algorithms for driver allocation.

### 5. How would you implement ride tracking?
**Answer**: Use WebSocket connections for real-time location updates. Implement location smoothing algorithms. Use GPS accuracy validation. Consider using map APIs for route visualization.

### 6. What happens if the payment fails?
**Answer**: Implement retry logic with exponential backoff. Use multiple payment methods as fallback. Implement payment hold and release mechanisms. Consider using escrow services for ride completion.

### 7. How do you handle driver earnings and payouts?
**Answer**: Implement daily/weekly payout schedules. Use escrow accounts for driver earnings. Implement earnings tracking and reporting. Consider using blockchain for transparent payouts.

### 8. What's your approach to ride safety?
**Answer**: Implement emergency button functionality. Use driver background verification. Implement ride monitoring and alerts. Consider using AI for safety risk assessment.

### 9. How would you implement ride sharing?
**Answer**: Use route optimization algorithms for multiple passengers. Implement passenger matching based on routes. Use dynamic pricing for shared rides. Consider using real-time route updates.

### 10. What's your strategy for handling ride disputes?
**Answer**: Implement ride recording and evidence collection. Use automated dispute resolution systems. Implement human review for complex cases. Consider using blockchain for immutable ride records.

### 11. How do you handle driver onboarding?
**Answer**: Implement multi-step verification process. Use document verification APIs. Implement driver training and certification. Consider using background check services.

### 12. What's your approach to ride pricing?
**Answer**: Implement dynamic pricing based on demand and supply. Use distance and time-based calculations. Implement surge pricing for peak hours. Consider using machine learning for price optimization.

### 13. How would you implement ride scheduling?
**Answer**: Use calendar-based scheduling system. Implement driver availability management. Use predictive algorithms for demand forecasting. Consider using time-based pricing for scheduled rides.

### 14. What's your strategy for handling ride cancellations?
**Answer**: Implement cancellation policies with time limits. Use cancellation fees for late cancellations. Implement driver compensation for cancellations. Consider using cancellation prediction models.

### 15. How do you handle ride quality and ratings?
**Answer**: Implement bidirectional rating system. Use rating aggregation and filtering. Implement rating-based driver incentives. Consider using machine learning for rating analysis.

### 16. What's your approach to ride analytics?
**Answer**: Implement real-time dashboards for ride metrics. Use data warehouse for historical analysis. Implement predictive analytics for demand forecasting. Consider using business intelligence tools.

### 17. How would you implement ride insurance?
**Answer**: Integrate with insurance providers for ride coverage. Implement claim processing and management. Use risk assessment for insurance pricing. Consider using blockchain for insurance claims.

### 18. What's your strategy for handling ride fraud?
**Answer**: Implement fraud detection algorithms. Use behavioral analysis for suspicious activities. Implement ride verification systems. Consider using machine learning for fraud prevention.

### 19. How do you handle ride accessibility?
**Answer**: Implement accessibility features for disabled users. Use specialized vehicle types for accessibility needs. Implement accessibility rating system. Consider using accessibility compliance standards.

### 20. What's your approach to ride sustainability?
**Answer**: Implement carbon footprint tracking for rides. Use electric vehicle incentives. Implement ride sharing for environmental benefits. Consider using sustainability metrics for driver rewards.

## Evaluation Checklist

### Code Quality (25%)
- [ ] Clean, readable Go code with proper error handling
- [ ] Appropriate use of interfaces and structs
- [ ] Proper concurrency patterns (goroutines, channels)
- [ ] Good separation of concerns

### Architecture (25%)
- [ ] Scalable design with geospatial indexing
- [ ] Proper ride state management
- [ ] Efficient driver matching algorithms
- [ ] Real-time update mechanisms

### Functionality (25%)
- [ ] User and driver registration working
- [ ] Ride booking and matching functional
- [ ] Payment processing implemented
- [ ] Rating system working

### Testing (15%)
- [ ] Unit tests for core functionality
- [ ] Integration tests for API endpoints
- [ ] Edge case testing
- [ ] Performance testing

### Discussion (10%)
- [ ] Clear explanation of design decisions
- [ ] Understanding of scaling challenges
- [ ] Knowledge of geospatial algorithms
- [ ] Ability to discuss trade-offs

## Discussion Pointers

### Key Points to Highlight
1. **Geospatial Indexing**: Explain the use of spatial data structures for efficient driver queries
2. **Ride State Management**: Discuss the state machine for ride lifecycle
3. **Driver Matching**: Explain the algorithm for finding nearest available drivers
4. **Real-time Updates**: Discuss WebSocket usage for live tracking
5. **Payment Integration**: Explain the payment processing flow

### Trade-offs to Discuss
1. **Accuracy vs Performance**: GPS accuracy vs response time trade-offs
2. **Fairness vs Efficiency**: Driver assignment fairness vs system efficiency
3. **Cost vs Quality**: Service cost vs ride quality trade-offs
4. **Privacy vs Safety**: User privacy vs safety monitoring trade-offs
5. **Centralization vs Decentralization**: Centralized vs decentralized driver management

### Extension Scenarios
1. **Multi-city Deployment**: How to handle geographic distribution
2. **Ride Sharing Integration**: Multiple passengers and route optimization
3. **Autonomous Vehicles**: Integration with self-driving cars
4. **Public Transport Integration**: Multi-modal transportation
5. **Enterprise Features**: Corporate accounts and bulk booking
