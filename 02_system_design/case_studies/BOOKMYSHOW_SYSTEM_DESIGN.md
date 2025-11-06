---
# Auto-generated front matter
Title: Bookmyshow System Design
LastUpdated: 2025-11-06T20:45:57.731699
Tags: []
Status: draft
---

# 🎬 BookMyShow System Design

> **Complete system design for a movie ticketing platform like BookMyShow**

## 📋 **Requirements Analysis**

### **Functional Requirements**

- User registration and authentication
- Browse movies, theaters, and shows
- Search and filter movies by location, genre, language
- Select seats and book tickets
- Payment processing
- Ticket confirmation and cancellation
- User reviews and ratings
- Admin panel for theaters and movies

### **Non-Functional Requirements**

- **Scalability**: Handle 10M+ users, 100K+ concurrent bookings
- **Availability**: 99.9% uptime
- **Performance**: <200ms response time for search, <500ms for booking
- **Consistency**: Strong consistency for seat booking, eventual for reviews
- **Reliability**: Handle peak loads during movie releases

### **Scale Estimation**

```
Daily Active Users: 10M
Peak Concurrent Users: 100K
Movies per day: 50K shows
Theaters: 10K across India
Seats per theater: 200-500
Peak booking rate: 10K bookings/minute
Data storage: 100TB+ (movies, shows, bookings, users)
```

---

## 🏗️ **High-Level Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │   Web App   │ │  Mobile App │ │  Admin App  │ │  Theater App││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                    Load Balancer                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │     LB1     │ │     LB2     │ │     LB3     │ │     LB4     ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                    API Gateway                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │  Auth &     │ │  Rate       │ │  Request    │ │  Response   ││
│  │  Authz      │ │  Limiting   │ │  Routing    │ │  Caching    ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                  Microservices Layer                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │   User      │ │   Movie     │ │  Theater    │ │  Booking    ││
│  │  Service    │ │  Service    │ │  Service    │ │  Service    ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │  Payment    │ │  Search     │ │  Review     │ │  Notification││
│  │  Service    │ │  Service    │ │  Service    │ │  Service    ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                    Data Layer                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │   MySQL     │ │    Redis    │ │   Elastic   │ │   MongoDB   ││
│  │  Cluster    │ │   Cluster   │ │   Search    │ │  (Reviews)  ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │   Kafka     │ │     S3      │ │   CDN       │ │   Monitoring││
│  │  (Events)   │ │ (Media)     │ │ (Static)    │ │   & Logging ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 **Detailed Component Design**

### **1. User Service**

```go
type UserService struct {
    db          *sql.DB
    cache       *redis.Client
    authClient  *AuthClient
    eventBus    *EventBus
}

type User struct {
    ID          string    `json:"id"`
    Email       string    `json:"email"`
    Name        string    `json:"name"`
    Phone       string    `json:"phone"`
    CreatedAt   time.Time `json:"created_at"`
    UpdatedAt   time.Time `json:"updated_at"`
    Preferences UserPreferences `json:"preferences"`
}

type UserPreferences struct {
    PreferredLanguage string   `json:"preferred_language"`
    PreferredGenre    []string `json:"preferred_genre"`
    PreferredLocation string   `json:"preferred_location"`
    NotificationSettings NotificationSettings `json:"notification_settings"`
}

func (us *UserService) RegisterUser(ctx context.Context, req *RegisterRequest) (*User, error) {
    // 1. Validate input
    if err := us.validateRegistration(req); err != nil {
        return nil, err
    }

    // 2. Check if user exists
    if exists, err := us.userExists(req.Email); err != nil {
        return nil, err
    } else if exists {
        return nil, ErrUserAlreadyExists
    }

    // 3. Hash password
    hashedPassword, err := us.hashPassword(req.Password)
    if err != nil {
        return nil, err
    }

    // 4. Create user
    user := &User{
        ID:        generateUUID(),
        Email:     req.Email,
        Name:      req.Name,
        Phone:     req.Phone,
        CreatedAt: time.Now(),
    }

    // 5. Save to database
    if err := us.saveUser(user, hashedPassword); err != nil {
        return nil, err
    }

    // 6. Cache user data
    us.cache.Set(fmt.Sprintf("user:%s", user.ID), user, time.Hour)

    // 7. Publish user created event
    us.eventBus.Publish("user.created", &UserCreatedEvent{
        UserID: user.ID,
        Email:  user.Email,
    })

    return user, nil
}

func (us *UserService) GetUser(ctx context.Context, userID string) (*User, error) {
    // 1. Check cache first
    if cached, err := us.cache.Get(fmt.Sprintf("user:%s", userID)).Result(); err == nil {
        var user User
        json.Unmarshal([]byte(cached), &user)
        return &user, nil
    }

    // 2. Get from database
    user, err := us.getUserFromDB(userID)
    if err != nil {
        return nil, err
    }

    // 3. Cache the result
    userJSON, _ := json.Marshal(user)
    us.cache.Set(fmt.Sprintf("user:%s", userID), userJSON, time.Hour)

    return user, nil
}
```

### **2. Movie Service**

```go
type MovieService struct {
    db          *sql.DB
    cache       *redis.Client
    searchIndex *elasticsearch.Client
    eventBus    *EventBus
}

type Movie struct {
    ID          string    `json:"id"`
    Title       string    `json:"title"`
    Description string    `json:"description"`
    Genre       []string  `json:"genre"`
    Language    string    `json:"language"`
    Duration    int       `json:"duration"` // in minutes
    ReleaseDate time.Time `json:"release_date"`
    Rating      float64   `json:"rating"`
    PosterURL   string    `json:"poster_url"`
    TrailerURL  string    `json:"trailer_url"`
    Cast        []string  `json:"cast"`
    Director    string    `json:"director"`
    Status      string    `json:"status"` // active, inactive, coming_soon
    CreatedAt   time.Time `json:"created_at"`
    UpdatedAt   time.Time `json:"updated_at"`
}

func (ms *MovieService) SearchMovies(ctx context.Context, req *SearchRequest) (*SearchResponse, error) {
    // 1. Build search query
    query := ms.buildSearchQuery(req)

    // 2. Search in Elasticsearch
    searchResult, err := ms.searchIndex.Search().
        Index("movies").
        Query(query).
        From(req.Offset).
        Size(req.Limit).
        Do(ctx)

    if err != nil {
        return nil, err
    }

    // 3. Extract movie IDs
    var movieIDs []string
    for _, hit := range searchResult.Hits.Hits {
        movieIDs = append(movieIDs, hit.Id)
    }

    // 4. Get movie details from cache/database
    movies, err := ms.getMoviesByIDs(movieIDs)
    if err != nil {
        return nil, err
    }

    return &SearchResponse{
        Movies: movies,
        Total:  searchResult.Hits.TotalHits.Value,
        Page:   req.Page,
        Limit:  req.Limit,
    }, nil
}

func (ms *MovieService) GetMovieDetails(ctx context.Context, movieID string) (*Movie, error) {
    // 1. Check cache first
    if cached, err := ms.cache.Get(fmt.Sprintf("movie:%s", movieID)).Result(); err == nil {
        var movie Movie
        json.Unmarshal([]byte(cached), &movie)
        return &movie, nil
    }

    // 2. Get from database
    movie, err := ms.getMovieFromDB(movieID)
    if err != nil {
        return nil, err
    }

    // 3. Cache the result
    movieJSON, _ := json.Marshal(movie)
    ms.cache.Set(fmt.Sprintf("movie:%s", movieID), movieJSON, time.Hour*24)

    return movie, nil
}

func (ms *MovieService) buildSearchQuery(req *SearchRequest) *elasticsearch.Query {
    var queries []elasticsearch.Query

    // Text search
    if req.Query != "" {
        queries = append(queries, elasticsearch.NewMultiMatchQuery(req.Query, "title", "description", "cast", "director"))
    }

    // Genre filter
    if len(req.Genres) > 0 {
        queries = append(queries, elasticsearch.NewTermsQuery("genre", req.Genres...))
    }

    // Language filter
    if req.Language != "" {
        queries = append(queries, elasticsearch.NewTermQuery("language", req.Language))
    }

    // Location filter (theaters showing the movie)
    if req.City != "" {
        queries = append(queries, elasticsearch.NewTermQuery("available_cities", req.City))
    }

    // Date filter
    if !req.Date.IsZero() {
        queries = append(queries, elasticsearch.NewRangeQuery("release_date").Lte(req.Date))
    }

    return elasticsearch.NewBoolQuery().Must(queries...)
}
```

### **3. Theater Service**

```go
type TheaterService struct {
    db          *sql.DB
    cache       *redis.Client
    eventBus    *EventBus
}

type Theater struct {
    ID          string    `json:"id"`
    Name        string    `json:"name"`
    Address     string    `json:"address"`
    City        string    `json:"city"`
    State       string    `json:"state"`
    Pincode     string    `json:"pincode"`
    Latitude    float64   `json:"latitude"`
    Longitude   float64   `json:"longitude"`
    Amenities   []string  `json:"amenities"`
    Screens     []Screen  `json:"screens"`
    CreatedAt   time.Time `json:"created_at"`
    UpdatedAt   time.Time `json:"updated_at"`
}

type Screen struct {
    ID       string `json:"id"`
    Name     string `json:"name"`
    Capacity int    `json:"capacity"`
    Type     string `json:"type"` // 2D, 3D, IMAX
    Seats    []Seat `json:"seats"`
}

type Seat struct {
    ID       string `json:"id"`
    Row      string `json:"row"`
    Number   int    `json:"number"`
    Type     string `json:"type"` // regular, premium, recliner
    Price    float64 `json:"price"`
    Status   string `json:"status"` // available, booked, blocked
}

func (ts *TheaterService) GetTheatersByCity(ctx context.Context, city string) ([]*Theater, error) {
    // 1. Check cache first
    cacheKey := fmt.Sprintf("theaters:city:%s", city)
    if cached, err := ts.cache.Get(cacheKey).Result(); err == nil {
        var theaters []*Theater
        json.Unmarshal([]byte(cached), &theaters)
        return theaters, nil
    }

    // 2. Get from database
    theaters, err := ts.getTheatersFromDB(city)
    if err != nil {
        return nil, err
    }

    // 3. Cache the result
    theatersJSON, _ := json.Marshal(theaters)
    ts.cache.Set(cacheKey, theatersJSON, time.Hour*6)

    return theaters, nil
}

func (ts *TheaterService) GetShowsByMovieAndTheater(ctx context.Context, movieID, theaterID string, date time.Time) ([]*Show, error) {
    // 1. Get shows from database
    shows, err := ts.getShowsFromDB(movieID, theaterID, date)
    if err != nil {
        return nil, err
    }

    // 2. Get seat availability for each show
    for _, show := range shows {
        availability, err := ts.getSeatAvailability(show.ID)
        if err != nil {
            return nil, err
        }
        show.AvailableSeats = availability
    }

    return shows, nil
}
```

### **4. Booking Service (Core Component)**

```go
type BookingService struct {
    db              *sql.DB
    cache           *redis.Client
    paymentService  *PaymentService
    notificationService *NotificationService
    eventBus        *EventBus
    seatLockManager *SeatLockManager
}

type Booking struct {
    ID          string    `json:"id"`
    UserID      string    `json:"user_id"`
    ShowID      string    `json:"show_id"`
    TheaterID   string    `json:"theater_id"`
    MovieID     string    `json:"movie_id"`
    Seats       []Seat    `json:"seats"`
    TotalAmount float64   `json:"total_amount"`
    Status      string    `json:"status"` // pending, confirmed, cancelled
    PaymentID   string    `json:"payment_id"`
    CreatedAt   time.Time `json:"created_at"`
    ExpiresAt   time.Time `json:"expires_at"`
}

type SeatLock struct {
    SeatID    string    `json:"seat_id"`
    UserID    string    `json:"user_id"`
    ShowID    string    `json:"show_id"`
    LockedAt  time.Time `json:"locked_at"`
    ExpiresAt time.Time `json:"expires_at"`
}

func (bs *BookingService) CreateBooking(ctx context.Context, req *CreateBookingRequest) (*Booking, error) {
    // 1. Validate request
    if err := bs.validateBookingRequest(req); err != nil {
        return nil, err
    }

    // 2. Lock seats (with timeout)
    lockID, err := bs.seatLockManager.LockSeats(req.ShowID, req.SeatIDs, req.UserID, 10*time.Minute)
    if err != nil {
        return nil, err
    }

    // 3. Create booking
    booking := &Booking{
        ID:          generateUUID(),
        UserID:      req.UserID,
        ShowID:      req.ShowID,
        TheaterID:   req.TheaterID,
        MovieID:     req.MovieID,
        Seats:       req.Seats,
        TotalAmount: req.TotalAmount,
        Status:      "pending",
        CreatedAt:   time.Now(),
        ExpiresAt:   time.Now().Add(10 * time.Minute),
    }

    // 4. Save booking to database
    if err := bs.saveBooking(booking); err != nil {
        bs.seatLockManager.UnlockSeats(lockID)
        return nil, err
    }

    // 5. Process payment
    paymentResult, err := bs.paymentService.ProcessPayment(ctx, &PaymentRequest{
        Amount:    booking.TotalAmount,
        UserID:    booking.UserID,
        BookingID: booking.ID,
    })
    if err != nil {
        bs.seatLockManager.UnlockSeats(lockID)
        booking.Status = "failed"
        bs.updateBooking(booking)
        return nil, err
    }

    // 6. Confirm booking
    booking.Status = "confirmed"
    booking.PaymentID = paymentResult.PaymentID
    if err := bs.updateBooking(booking); err != nil {
        return nil, err
    }

    // 7. Release seat locks
    bs.seatLockManager.UnlockSeats(lockID)

    // 8. Send confirmation notification
    bs.notificationService.SendBookingConfirmation(booking)

    // 9. Publish booking confirmed event
    bs.eventBus.Publish("booking.confirmed", &BookingConfirmedEvent{
        BookingID: booking.ID,
        UserID:    booking.UserID,
        MovieID:   booking.MovieID,
        TheaterID: booking.TheaterID,
    })

    return booking, nil
}

func (bs *BookingService) CancelBooking(ctx context.Context, bookingID string) error {
    // 1. Get booking
    booking, err := bs.getBooking(bookingID)
    if err != nil {
        return err
    }

    // 2. Check if cancellation is allowed
    if !bs.canCancelBooking(booking) {
        return ErrCancellationNotAllowed
    }

    // 3. Process refund
    if err := bs.paymentService.ProcessRefund(booking.PaymentID); err != nil {
        return err
    }

    // 4. Update booking status
    booking.Status = "cancelled"
    if err := bs.updateBooking(booking); err != nil {
        return err
    }

    // 5. Release seats
    for _, seat := range booking.Seats {
        bs.releaseSeat(seat.ID, booking.ShowID)
    }

    // 6. Send cancellation notification
    bs.notificationService.SendBookingCancellation(booking)

    return nil
}
```

### **5. Seat Lock Manager**

```go
type SeatLockManager struct {
    cache *redis.Client
    mutex sync.RWMutex
}

func (slm *SeatLockManager) LockSeats(showID string, seatIDs []string, userID string, duration time.Duration) (string, error) {
    lockID := generateUUID()
    lockKey := fmt.Sprintf("seat_lock:%s", lockID)

    // Create seat lock data
    seatLock := &SeatLock{
        SeatID:    strings.Join(seatIDs, ","),
        UserID:    userID,
        ShowID:    showID,
        LockedAt:  time.Now(),
        ExpiresAt: time.Now().Add(duration),
    }

    // Store in Redis with expiration
    seatLockJSON, _ := json.Marshal(seatLock)
    if err := slm.cache.Set(lockKey, seatLockJSON, duration).Err(); err != nil {
        return "", err
    }

    // Mark seats as locked
    for _, seatID := range seatIDs {
        seatKey := fmt.Sprintf("seat:locked:%s:%s", showID, seatID)
        slm.cache.Set(seatKey, userID, duration)
    }

    return lockID, nil
}

func (slm *SeatLockManager) UnlockSeats(lockID string) error {
    lockKey := fmt.Sprintf("seat_lock:%s", lockID)

    // Get seat lock data
    seatLockJSON, err := slm.cache.Get(lockKey).Result()
    if err != nil {
        return err
    }

    var seatLock SeatLock
    json.Unmarshal([]byte(seatLockJSON), &seatLock)

    // Release individual seat locks
    seatIDs := strings.Split(seatLock.SeatID, ",")
    for _, seatID := range seatIDs {
        seatKey := fmt.Sprintf("seat:locked:%s:%s", seatLock.ShowID, seatID)
        slm.cache.Del(seatKey)
    }

    // Remove the main lock
    slm.cache.Del(lockKey)

    return nil
}

func (slm *SeatLockManager) IsSeatLocked(showID, seatID string) (bool, string) {
    seatKey := fmt.Sprintf("seat:locked:%s:%s", showID, seatID)
    userID, err := slm.cache.Get(seatKey).Result()
    if err != nil {
        return false, ""
    }
    return true, userID
}
```

---

## 🗄️ **Database Design**

### **1. MySQL Schema**

```sql
-- Users table
CREATE TABLE users (
    id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    phone VARCHAR(20),
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_email (email),
    INDEX idx_phone (phone)
);

-- Movies table
CREATE TABLE movies (
    id VARCHAR(36) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    genre JSON,
    language VARCHAR(50),
    duration INT,
    release_date DATE,
    rating DECIMAL(3,1),
    poster_url VARCHAR(500),
    trailer_url VARCHAR(500),
    cast JSON,
    director VARCHAR(255),
    status ENUM('active', 'inactive', 'coming_soon') DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_title (title),
    INDEX idx_genre (genre),
    INDEX idx_language (language),
    INDEX idx_release_date (release_date),
    INDEX idx_status (status)
);

-- Theaters table
CREATE TABLE theaters (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    address TEXT,
    city VARCHAR(100),
    state VARCHAR(100),
    pincode VARCHAR(10),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    amenities JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_city (city),
    INDEX idx_location (latitude, longitude)
);

-- Screens table
CREATE TABLE screens (
    id VARCHAR(36) PRIMARY KEY,
    theater_id VARCHAR(36),
    name VARCHAR(100),
    capacity INT,
    type ENUM('2D', '3D', 'IMAX'),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (theater_id) REFERENCES theaters(id),
    INDEX idx_theater_id (theater_id)
);

-- Seats table
CREATE TABLE seats (
    id VARCHAR(36) PRIMARY KEY,
    screen_id VARCHAR(36),
    row_name VARCHAR(10),
    seat_number INT,
    type ENUM('regular', 'premium', 'recliner'),
    price DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (screen_id) REFERENCES screens(id),
    INDEX idx_screen_id (screen_id),
    INDEX idx_row_seat (screen_id, row_name, seat_number)
);

-- Shows table
CREATE TABLE shows (
    id VARCHAR(36) PRIMARY KEY,
    movie_id VARCHAR(36),
    theater_id VARCHAR(36),
    screen_id VARCHAR(36),
    show_time DATETIME,
    price DECIMAL(10,2),
    status ENUM('active', 'inactive', 'cancelled') DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (movie_id) REFERENCES movies(id),
    FOREIGN KEY (theater_id) REFERENCES theaters(id),
    FOREIGN KEY (screen_id) REFERENCES screens(id),
    INDEX idx_movie_theater (movie_id, theater_id),
    INDEX idx_show_time (show_time),
    INDEX idx_status (status)
);

-- Bookings table
CREATE TABLE bookings (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36),
    show_id VARCHAR(36),
    theater_id VARCHAR(36),
    movie_id VARCHAR(36),
    total_amount DECIMAL(10,2),
    status ENUM('pending', 'confirmed', 'cancelled', 'failed') DEFAULT 'pending',
    payment_id VARCHAR(36),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (show_id) REFERENCES shows(id),
    FOREIGN KEY (theater_id) REFERENCES theaters(id),
    FOREIGN KEY (movie_id) REFERENCES movies(id),
    INDEX idx_user_id (user_id),
    INDEX idx_show_id (show_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
);

-- Booking seats table
CREATE TABLE booking_seats (
    id VARCHAR(36) PRIMARY KEY,
    booking_id VARCHAR(36),
    seat_id VARCHAR(36),
    price DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (booking_id) REFERENCES bookings(id),
    FOREIGN KEY (seat_id) REFERENCES seats(id),
    INDEX idx_booking_id (booking_id),
    INDEX idx_seat_id (seat_id)
);
```

### **2. Redis Cache Strategy**

```go
// Cache keys and TTL
const (
    // User cache
    UserCacheKey = "user:%s"           // TTL: 1 hour
    UserSessionKey = "session:%s"      // TTL: 24 hours

    // Movie cache
    MovieCacheKey = "movie:%s"         // TTL: 24 hours
    MovieSearchKey = "search:%s"       // TTL: 1 hour

    // Theater cache
    TheaterCacheKey = "theater:%s"     // TTL: 6 hours
    TheaterCityKey = "theaters:city:%s" // TTL: 6 hours

    // Show cache
    ShowCacheKey = "show:%s"           // TTL: 1 hour
    ShowSeatsKey = "show:seats:%s"     // TTL: 5 minutes

    // Seat locks
    SeatLockKey = "seat:locked:%s:%s"  // TTL: 10 minutes
    SeatLockMainKey = "seat_lock:%s"   // TTL: 10 minutes
)
```

---

## 🔍 **Search and Filtering**

### **Elasticsearch Index**

```json
{
  "mappings": {
    "properties": {
      "id": { "type": "keyword" },
      "title": {
        "type": "text",
        "analyzer": "standard",
        "fields": {
          "suggest": {
            "type": "completion"
          }
        }
      },
      "description": { "type": "text" },
      "genre": { "type": "keyword" },
      "language": { "type": "keyword" },
      "duration": { "type": "integer" },
      "release_date": { "type": "date" },
      "rating": { "type": "float" },
      "cast": { "type": "text" },
      "director": { "type": "text" },
      "status": { "type": "keyword" },
      "available_cities": { "type": "keyword" },
      "theater_count": { "type": "integer" },
      "show_count": { "type": "integer" }
    }
  }
}
```

### **Search Implementation**

```go
func (ms *MovieService) SearchMovies(ctx context.Context, req *SearchRequest) (*SearchResponse, error) {
    // Build complex search query
    boolQuery := elastic.NewBoolQuery()

    // Text search with multiple fields
    if req.Query != "" {
        multiMatchQuery := elastic.NewMultiMatchQuery(req.Query).
            Fields("title^3", "description", "cast^2", "director^2").
            Type("best_fields").
            Fuzziness("AUTO")
        boolQuery.Must(multiMatchQuery)
    }

    // Genre filter
    if len(req.Genres) > 0 {
        genreQuery := elastic.NewTermsQuery("genre", req.Genres...)
        boolQuery.Filter(genreQuery)
    }

    // Language filter
    if req.Language != "" {
        languageQuery := elastic.NewTermQuery("language", req.Language)
        boolQuery.Filter(languageQuery)
    }

    // City filter (movies available in specific city)
    if req.City != "" {
        cityQuery := elastic.NewTermQuery("available_cities", req.City)
        boolQuery.Filter(cityQuery)
    }

    // Date range filter
    if !req.Date.IsZero() {
        dateQuery := elastic.NewRangeQuery("release_date").Lte(req.Date)
        boolQuery.Filter(dateQuery)
    }

    // Rating filter
    if req.MinRating > 0 {
        ratingQuery := elastic.NewRangeQuery("rating").Gte(req.MinRating)
        boolQuery.Filter(ratingQuery)
    }

    // Execute search
    searchResult, err := ms.searchIndex.Search().
        Index("movies").
        Query(boolQuery).
        Sort("rating", false). // Sort by rating descending
        From(req.Offset).
        Size(req.Limit).
        Highlight(elastic.NewHighlight().Field("title").Field("description")).
        Do(ctx)

    if err != nil {
        return nil, err
    }

    // Process results
    var movies []*Movie
    for _, hit := range searchResult.Hits.Hits {
        var movie Movie
        json.Unmarshal(hit.Source, &movie)
        movies = append(movies, &movie)
    }

    return &SearchResponse{
        Movies: movies,
        Total:  searchResult.Hits.TotalHits.Value,
        Page:   req.Page,
        Limit:  req.Limit,
    }, nil
}
```

---

## 💳 **Payment Integration**

### **Payment Service**

```go
type PaymentService struct {
    paymentGateway PaymentGateway
    db            *sql.DB
    cache         *redis.Client
    eventBus      *EventBus
}

type PaymentRequest struct {
    Amount    float64 `json:"amount"`
    UserID    string  `json:"user_id"`
    BookingID string  `json:"booking_id"`
    Currency  string  `json:"currency"`
}

type PaymentResponse struct {
    PaymentID     string `json:"payment_id"`
    Status        string `json:"status"`
    TransactionID string `json:"transaction_id"`
    GatewayRef    string `json:"gateway_ref"`
}

func (ps *PaymentService) ProcessPayment(ctx context.Context, req *PaymentRequest) (*PaymentResponse, error) {
    // 1. Create payment record
    payment := &Payment{
        ID:        generateUUID(),
        UserID:    req.UserID,
        BookingID: req.BookingID,
        Amount:    req.Amount,
        Currency:  req.Currency,
        Status:    "pending",
        CreatedAt: time.Now(),
    }

    if err := ps.savePayment(payment); err != nil {
        return nil, err
    }

    // 2. Process with payment gateway
    gatewayResp, err := ps.paymentGateway.ProcessPayment(ctx, &GatewayRequest{
        Amount:    req.Amount,
        Currency:  req.Currency,
        UserID:    req.UserID,
        PaymentID: payment.ID,
    })
    if err != nil {
        payment.Status = "failed"
        ps.updatePayment(payment)
        return nil, err
    }

    // 3. Update payment status
    payment.Status = gatewayResp.Status
    payment.TransactionID = gatewayResp.TransactionID
    payment.GatewayRef = gatewayResp.GatewayRef

    if err := ps.updatePayment(payment); err != nil {
        return nil, err
    }

    // 4. Publish payment event
    ps.eventBus.Publish("payment.processed", &PaymentProcessedEvent{
        PaymentID: payment.ID,
        BookingID: req.BookingID,
        Status:    payment.Status,
        Amount:    payment.Amount,
    })

    return &PaymentResponse{
        PaymentID:     payment.ID,
        Status:        payment.Status,
        TransactionID: payment.TransactionID,
        GatewayRef:    payment.GatewayRef,
    }, nil
}
```

---

## 📱 **Real-time Features**

### **WebSocket for Live Updates**

```go
type WebSocketManager struct {
    clients    map[string]*websocket.Conn
    register   chan *Client
    unregister chan *Client
    broadcast  chan []byte
    mutex      sync.RWMutex
}

type Client struct {
    ID     string
    UserID string
    Conn   *websocket.Conn
    Send   chan []byte
}

func (wsm *WebSocketManager) HandleClient(client *Client) {
    defer func() {
        wsm.unregister <- client
        client.Conn.Close()
    }()

    for {
        select {
        case message := <-client.Send:
            if err := client.Conn.WriteMessage(websocket.TextMessage, message); err != nil {
                return
            }
        }
    }
}

func (wsm *WebSocketManager) BroadcastSeatUpdate(showID string, seatID string, status string) {
    message := &SeatUpdateMessage{
        Type:    "seat_update",
        ShowID:  showID,
        SeatID:  seatID,
        Status:  status,
        Time:    time.Now(),
    }

    messageJSON, _ := json.Marshal(message)
    wsm.broadcast <- messageJSON
}
```

---

## 📊 **Monitoring and Analytics**

### **Metrics Collection**

```go
type MetricsCollector struct {
    prometheus *prometheus.Registry
    logger     *logrus.Logger
}

func (mc *MetricsCollector) TrackBookingMetrics(booking *Booking) {
    // Track booking metrics
    mc.prometheus.GetCounter("bookings_total").
        WithLabelValues(booking.Status, booking.MovieID, booking.TheaterID).
        Inc()

    // Track revenue
    mc.prometheus.GetCounter("revenue_total").
        WithLabelValues(booking.MovieID, booking.TheaterID).
        Add(booking.TotalAmount)

    // Track booking latency
    mc.prometheus.GetHistogram("booking_duration_seconds").
        WithLabelValues("create_booking").
        Observe(time.Since(booking.CreatedAt).Seconds())
}

func (mc *MetricsCollector) TrackSearchMetrics(query string, resultCount int, latency time.Duration) {
    // Track search metrics
    mc.prometheus.GetCounter("searches_total").
        WithLabelValues("movie_search").
        Inc()

    mc.prometheus.GetHistogram("search_duration_seconds").
        WithLabelValues("movie_search").
        Observe(latency.Seconds())

    mc.prometheus.GetHistogram("search_results_count").
        WithLabelValues("movie_search").
        Observe(float64(resultCount))
}
```

---

## 🚀 **Scalability Considerations**

### **1. Database Sharding**

```go
type ShardingStrategy struct {
    shards map[string]*sql.DB
    hashFunc func(string) string
}

func (ss *ShardingStrategy) GetShard(bookingID string) *sql.DB {
    // Shard by booking ID for even distribution
    shardKey := ss.hashFunc(bookingID)
    return ss.shards[shardKey]
}

func (ss *ShardingStrategy) GetShardByUser(userID string) *sql.DB {
    // Shard by user ID for user-specific queries
    shardKey := ss.hashFunc(userID)
    return ss.shards[shardKey]
}
```

### **2. Caching Strategy**

```go
type CacheStrategy struct {
    l1Cache *sync.Map        // In-memory cache (1ms)
    l2Cache *redis.Client    // Redis cache (5ms)
    l3Cache *DatabaseCache   // Database cache (50ms)
}

func (cs *CacheStrategy) Get(key string) (interface{}, error) {
    // L1 cache lookup
    if value, ok := cs.l1Cache.Load(key); ok {
        return value, nil
    }

    // L2 cache lookup
    if value, err := cs.l2Cache.Get(key).Result(); err == nil {
        cs.l1Cache.Store(key, value)
        return value, nil
    }

    // L3 cache lookup
    if value, err := cs.l3Cache.Get(key); err == nil {
        cs.l2Cache.Set(key, value, time.Hour)
        cs.l1Cache.Store(key, value)
        return value, nil
    }

    return nil, ErrNotFound
}
```

### **3. Load Balancing**

```go
type LoadBalancer struct {
    servers []*Server
    strategy LoadBalancingStrategy
    healthChecker *HealthChecker
}

type LoadBalancingStrategy interface {
    SelectServer(servers []*Server, request *Request) *Server
}

type RoundRobinStrategy struct {
    current int
    mutex   sync.Mutex
}

func (rr *RoundRobinStrategy) SelectServer(servers []*Server, request *Request) *Server {
    rr.mutex.Lock()
    defer rr.mutex.Unlock()

    server := servers[rr.current]
    rr.current = (rr.current + 1) % len(servers)
    return server
}
```

---

## 🎯 **Key Design Decisions**

### **1. Why Microservices?**

- **Scalability**: Each service can scale independently
- **Technology Diversity**: Different services can use different tech stacks
- **Fault Isolation**: Failure in one service doesn't affect others
- **Team Autonomy**: Different teams can work on different services

### **2. Why Event-Driven Architecture?**

- **Decoupling**: Services communicate through events
- **Scalability**: Event processing can be scaled independently
- **Reliability**: Events can be retried and persisted
- **Flexibility**: Easy to add new consumers

### **3. Why Redis for Caching?**

- **Performance**: Sub-millisecond response times
- **Data Structures**: Rich data types for complex caching needs
- **Persistence**: Can persist data to disk
- **Clustering**: Built-in clustering support

### **4. Why Elasticsearch for Search?**

- **Full-text Search**: Advanced search capabilities
- **Faceted Search**: Multiple filters and aggregations
- **Real-time**: Near real-time search results
- **Scalability**: Horizontal scaling support

### **5. Why Seat Locking?**

- **Concurrency**: Prevent double booking of seats
- **User Experience**: Reserve seats during booking process
- **Timeout**: Automatic release of locked seats
- **Fairness**: First-come-first-served booking

---

## 🔒 **Security Considerations**

### **1. Authentication & Authorization**

```go
type AuthService struct {
    jwtSecret string
    db        *sql.DB
    cache     *redis.Client
}

func (as *AuthService) GenerateToken(userID string) (string, error) {
    claims := jwt.MapClaims{
        "user_id": userID,
        "exp":     time.Now().Add(time.Hour * 24).Unix(),
        "iat":     time.Now().Unix(),
    }

    token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    return token.SignedString([]byte(as.jwtSecret))
}

func (as *AuthService) ValidateToken(tokenString string) (*Claims, error) {
    token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
        return []byte(as.jwtSecret), nil
    })

    if err != nil {
        return nil, err
    }

    if claims, ok := token.Claims.(jwt.MapClaims); ok && token.Valid {
        return &Claims{
            UserID: claims["user_id"].(string),
            Exp:    int64(claims["exp"].(float64)),
        }, nil
    }

    return nil, ErrInvalidToken
}
```

### **2. Rate Limiting**

```go
type RateLimiter struct {
    cache *redis.Client
}

func (rl *RateLimiter) IsAllowed(userID string, limit int, window time.Duration) bool {
    key := fmt.Sprintf("rate_limit:%s", userID)

    // Use sliding window counter
    current := rl.cache.Incr(key).Val()
    if current == 1 {
        rl.cache.Expire(key, window)
    }

    return current <= int64(limit)
}
```

---

## 📈 **Performance Optimization**

### **1. Database Optimization**

- **Indexing**: Proper indexes on frequently queried columns
- **Query Optimization**: Efficient SQL queries
- **Connection Pooling**: Reuse database connections
- **Read Replicas**: Distribute read load

### **2. Caching Optimization**

- **Multi-level Caching**: L1 (memory), L2 (Redis), L3 (database)
- **Cache Warming**: Pre-populate frequently accessed data
- **Cache Invalidation**: Smart invalidation strategies
- **CDN**: Cache static content globally

### **3. Application Optimization**

- **Async Processing**: Non-blocking operations
- **Connection Pooling**: Reuse expensive connections
- **Object Pooling**: Reuse expensive objects
- **Goroutine Pools**: Limit concurrent goroutines

---

## 🎯 **Interview Success Tips**

### **1. Start with Requirements**

- Clarify functional and non-functional requirements
- Ask about scale, performance, and availability needs
- Understand user personas and use cases

### **2. Think in Layers**

- Client Layer (Web, Mobile, Admin)
- Load Balancer Layer
- API Gateway Layer
- Microservices Layer
- Data Layer

### **3. Discuss Trade-offs**

- Consistency vs Availability
- Performance vs Complexity
- Cost vs Scalability
- Development Speed vs Maintainability

### **4. Consider Edge Cases**

- Peak load during movie releases
- Network failures and timeouts
- Database failures and recovery
- Payment failures and rollbacks

### **5. Show Deep Thinking**

- Explain why you chose specific technologies
- Discuss alternative approaches
- Consider future scalability needs
- Address security and compliance requirements

---

**🎉 This comprehensive BookMyShow system design covers all the essential aspects for a Round 2 interview. Practice explaining each component and be ready to dive deeper into any area the interviewer is interested in! 🚀**
