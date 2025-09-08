# 🔐 Authentication: JWT, OAuth2, and Session Management

> **Master authentication mechanisms for secure backend systems**

## 📚 Concept

Authentication is the process of verifying the identity of a user, device, or system. It's the foundation of security in web applications and APIs.

### Authentication vs Authorization

- **Authentication**: "Who are you?" - Verifying identity
- **Authorization**: "What can you do?" - Determining permissions

### Common Authentication Methods

1. **Session-based**: Server-side session storage
2. **Token-based**: JWT, OAuth2, API keys
3. **Multi-factor**: SMS, TOTP, biometrics
4. **Federated**: SAML, OAuth2, OpenID Connect

## 🏗️ Authentication Architecture

```
Client ──Credentials──► Auth Service ──► Identity Provider
   │                        │                    │
   │                        ▼                    │
   │                   Session/Token             │
   │                        │                    │
   │                        ▼                    │
   └──Token/Session──► Protected Resources ◄─────┘
```

## 🛠️ Hands-on Example

### JWT Implementation (Go)

```go
package main

import (
    "crypto/rand"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "strings"
    "time"

    "github.com/golang-jwt/jwt/v5"
    "golang.org/x/crypto/bcrypt"
)

type User struct {
    ID       int    `json:"id"`
    Username string `json:"username"`
    Email    string `json:"email"`
    Password string `json:"-"`
    Role     string `json:"role"`
}

type Claims struct {
    UserID   int    `json:"user_id"`
    Username string `json:"username"`
    Role     string `json:"role"`
    jwt.RegisteredClaims
}

type AuthService struct {
    users      map[string]User
    jwtSecret  []byte
    sessions   map[string]time.Time
}

func NewAuthService() *AuthService {
    // Generate a random secret key
    secret := make([]byte, 32)
    rand.Read(secret)

    return &AuthService{
        users: map[string]User{
            "admin": {
                ID:       1,
                Username: "admin",
                Email:    "admin@example.com",
                Password: hashPassword("admin123"),
                Role:     "admin",
            },
            "user": {
                ID:       2,
                Username: "user",
                Email:    "user@example.com",
                Password: hashPassword("user123"),
                Role:     "user",
            },
        },
        jwtSecret: secret,
        sessions:  make(map[string]time.Time),
    }
}

func hashPassword(password string) string {
    hash, _ := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
    return string(hash)
}

func checkPassword(password, hash string) bool {
    err := bcrypt.CompareHashAndPassword([]byte(hash), []byte(password))
    return err == nil
}

func (as *AuthService) Login(w http.ResponseWriter, r *http.Request) {
    var credentials struct {
        Username string `json:"username"`
        Password string `json:"password"`
    }

    if err := json.NewDecoder(r.Body).Decode(&credentials); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }

    user, exists := as.users[credentials.Username]
    if !exists || !checkPassword(credentials.Password, user.Password) {
        http.Error(w, "Invalid credentials", http.StatusUnauthorized)
        return
    }

    // Generate JWT token
    token, err := as.generateJWT(user)
    if err != nil {
        http.Error(w, "Failed to generate token", http.StatusInternalServerError)
        return
    }

    // Generate refresh token
    refreshToken := as.generateRefreshToken()
    as.sessions[refreshToken] = time.Now().Add(7 * 24 * time.Hour)

    response := map[string]interface{}{
        "access_token":  token,
        "refresh_token": refreshToken,
        "token_type":    "Bearer",
        "expires_in":    3600, // 1 hour
        "user": map[string]interface{}{
            "id":       user.ID,
            "username": user.Username,
            "email":    user.Email,
            "role":     user.Role,
        },
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func (as *AuthService) generateJWT(user User) (string, error) {
    claims := Claims{
        UserID:   user.ID,
        Username: user.Username,
        Role:     user.Role,
        RegisteredClaims: jwt.RegisteredClaims{
            ExpiresAt: jwt.NewNumericDate(time.Now().Add(1 * time.Hour)),
            IssuedAt:  jwt.NewNumericDate(time.Now()),
            NotBefore: jwt.NewNumericDate(time.Now()),
            Issuer:    "auth-service",
            Subject:   fmt.Sprintf("%d", user.ID),
        },
    }

    token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    return token.SignedString(as.jwtSecret)
}

func (as *AuthService) generateRefreshToken() string {
    bytes := make([]byte, 32)
    rand.Read(bytes)
    return base64.URLEncoding.EncodeToString(bytes)
}

func (as *AuthService) RefreshToken(w http.ResponseWriter, r *http.Request) {
    var req struct {
        RefreshToken string `json:"refresh_token"`
    }

    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }

    // Check if refresh token exists and is valid
    expiresAt, exists := as.sessions[req.RefreshToken]
    if !exists || time.Now().After(expiresAt) {
        http.Error(w, "Invalid refresh token", http.StatusUnauthorized)
        return
    }

    // Get user from token (in real app, store user ID in refresh token)
    // For simplicity, we'll use the first user
    var user User
    for _, u := range as.users {
        user = u
        break
    }

    // Generate new access token
    token, err := as.generateJWT(user)
    if err != nil {
        http.Error(w, "Failed to generate token", http.StatusInternalServerError)
        return
    }

    response := map[string]interface{}{
        "access_token": token,
        "token_type":   "Bearer",
        "expires_in":   3600,
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func (as *AuthService) ValidateToken(tokenString string) (*Claims, error) {
    token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
        if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
            return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
        }
        return as.jwtSecret, nil
    })

    if err != nil {
        return nil, err
    }

    if claims, ok := token.Claims.(*Claims); ok && token.Valid {
        return claims, nil
    }

    return nil, fmt.Errorf("invalid token")
}

func (as *AuthService) AuthMiddleware(next http.HandlerFunc) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        authHeader := r.Header.Get("Authorization")
        if authHeader == "" {
            http.Error(w, "Authorization header required", http.StatusUnauthorized)
            return
        }

        // Extract token from "Bearer <token>"
        parts := strings.Split(authHeader, " ")
        if len(parts) != 2 || parts[0] != "Bearer" {
            http.Error(w, "Invalid authorization header format", http.StatusUnauthorized)
            return
        }

        token := parts[1]
        claims, err := as.ValidateToken(token)
        if err != nil {
            http.Error(w, "Invalid token", http.StatusUnauthorized)
            return
        }

        // Add user info to request context
        ctx := context.WithValue(r.Context(), "user", claims)
        next(w, r.WithContext(ctx))
    }
}

func (as *AuthService) RequireRole(role string) func(http.HandlerFunc) http.HandlerFunc {
    return func(next http.HandlerFunc) http.HandlerFunc {
        return as.AuthMiddleware(func(w http.ResponseWriter, r *http.Request) {
            claims := r.Context().Value("user").(*Claims)
            if claims.Role != role {
                http.Error(w, "Insufficient permissions", http.StatusForbidden)
                return
            }
            next(w, r)
        })
    }
}

// Protected endpoint
func (as *AuthService) GetProfile(w http.ResponseWriter, r *http.Request) {
    claims := r.Context().Value("user").(*Claims)

    user, exists := as.users[claims.Username]
    if !exists {
        http.Error(w, "User not found", http.StatusNotFound)
        return
    }

    // Remove password from response
    user.Password = ""

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

// Admin-only endpoint
func (as *AuthService) GetUsers(w http.ResponseWriter, r *http.Request) {
    users := make([]User, 0, len(as.users))
    for _, user := range as.users {
        user.Password = "" // Remove password
        users = append(users, user)
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]interface{}{
        "users": users,
        "total": len(users),
    })
}

func main() {
    authService := NewAuthService()

    // Public endpoints
    http.HandleFunc("/login", authService.Login)
    http.HandleFunc("/refresh", authService.RefreshToken)

    // Protected endpoints
    http.HandleFunc("/profile", authService.AuthMiddleware(authService.GetProfile))
    http.HandleFunc("/users", authService.RequireRole("admin")(authService.GetUsers))

    log.Println("Auth server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

### OAuth2 Implementation

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "net/url"
    "strings"
    "time"

    "golang.org/x/oauth2"
    "golang.org/x/oauth2/google"
)

type OAuth2Service struct {
    googleConfig *oauth2.Config
    users        map[string]User
}

func NewOAuth2Service() *OAuth2Service {
    return &OAuth2Service{
        googleConfig: &oauth2.Config{
            ClientID:     "your-google-client-id",
            ClientSecret: "your-google-client-secret",
            RedirectURL:  "http://localhost:8080/auth/google/callback",
            Scopes: []string{
                "https://www.googleapis.com/auth/userinfo.email",
                "https://www.googleapis.com/auth/userinfo.profile",
            },
            Endpoint: google.Endpoint,
        },
        users: make(map[string]User),
    }
}

func (os *OAuth2Service) GoogleLogin(w http.ResponseWriter, r *http.Request) {
    // Generate state parameter for CSRF protection
    state := generateRandomString(32)

    // Store state in session (in production, use secure session storage)
    http.SetCookie(w, &http.Cookie{
        Name:     "oauth_state",
        Value:    state,
        HttpOnly: true,
        Secure:   true,
        SameSite: http.SameSiteLaxMode,
    })

    // Redirect to Google OAuth
    authURL := os.googleConfig.AuthCodeURL(state, oauth2.AccessTypeOffline)
    http.Redirect(w, r, authURL, http.StatusTemporaryRedirect)
}

func (os *OAuth2Service) GoogleCallback(w http.ResponseWriter, r *http.Request) {
    // Verify state parameter
    state := r.URL.Query().Get("state")
    cookie, err := r.Cookie("oauth_state")
    if err != nil || state != cookie.Value {
        http.Error(w, "Invalid state parameter", http.StatusBadRequest)
        return
    }

    // Exchange code for token
    code := r.URL.Query().Get("code")
    token, err := os.googleConfig.Exchange(context.Background(), code)
    if err != nil {
        http.Error(w, "Failed to exchange code for token", http.StatusInternalServerError)
        return
    }

    // Get user info from Google
    userInfo, err := os.getGoogleUserInfo(token.AccessToken)
    if err != nil {
        http.Error(w, "Failed to get user info", http.StatusInternalServerError)
        return
    }

    // Create or update user
    user := User{
        ID:       len(os.users) + 1,
        Username: userInfo.Email,
        Email:    userInfo.Email,
        Role:     "user",
    }
    os.users[user.Email] = user

    // Generate JWT token
    jwtToken, err := os.generateJWT(user)
    if err != nil {
        http.Error(w, "Failed to generate JWT", http.StatusInternalServerError)
        return
    }

    // Redirect to frontend with token
    redirectURL := fmt.Sprintf("http://localhost:3000/auth/success?token=%s", jwtToken)
    http.Redirect(w, r, redirectURL, http.StatusTemporaryRedirect)
}

type GoogleUserInfo struct {
    ID    string `json:"id"`
    Email string `json:"email"`
    Name  string `json:"name"`
}

func (os *OAuth2Service) getGoogleUserInfo(accessToken string) (*GoogleUserInfo, error) {
    resp, err := http.Get("https://www.googleapis.com/oauth2/v2/userinfo?access_token=" + accessToken)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var userInfo GoogleUserInfo
    if err := json.NewDecoder(resp.Body).Decode(&userInfo); err != nil {
        return nil, err
    }

    return &userInfo, nil
}
```

### Session-based Authentication

```go
package main

import (
    "crypto/rand"
    "encoding/base64"
    "net/http"
    "sync"
    "time"
)

type SessionStore struct {
    sessions map[string]Session
    mutex    sync.RWMutex
}

type Session struct {
    UserID    int
    Username  string
    Role      string
    ExpiresAt time.Time
}

func NewSessionStore() *SessionStore {
    return &SessionStore{
        sessions: make(map[string]Session),
    }
}

func (ss *SessionStore) CreateSession(user User) string {
    sessionID := generateSessionID()

    ss.mutex.Lock()
    ss.sessions[sessionID] = Session{
        UserID:    user.ID,
        Username:  user.Username,
        Role:      user.Role,
        ExpiresAt: time.Now().Add(24 * time.Hour),
    }
    ss.mutex.Unlock()

    return sessionID
}

func (ss *SessionStore) GetSession(sessionID string) (*Session, bool) {
    ss.mutex.RLock()
    session, exists := ss.sessions[sessionID]
    ss.mutex.RUnlock()

    if !exists || time.Now().After(session.ExpiresAt) {
        return nil, false
    }

    return &session, true
}

func (ss *SessionStore) DeleteSession(sessionID string) {
    ss.mutex.Lock()
    delete(ss.sessions, sessionID)
    ss.mutex.Unlock()
}

func generateSessionID() string {
    bytes := make([]byte, 32)
    rand.Read(bytes)
    return base64.URLEncoding.EncodeToString(bytes)
}

func (as *AuthService) SessionLogin(w http.ResponseWriter, r *http.Request) {
    var credentials struct {
        Username string `json:"username"`
        Password string `json:"password"`
    }

    if err := json.NewDecoder(r.Body).Decode(&credentials); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }

    user, exists := as.users[credentials.Username]
    if !exists || !checkPassword(credentials.Password, user.Password) {
        http.Error(w, "Invalid credentials", http.StatusUnauthorized)
        return
    }

    // Create session
    sessionID := as.sessionStore.CreateSession(user)

    // Set session cookie
    http.SetCookie(w, &http.Cookie{
        Name:     "session_id",
        Value:    sessionID,
        HttpOnly: true,
        Secure:   true,
        SameSite: http.SameSiteLaxMode,
        MaxAge:   86400, // 24 hours
    })

    w.WriteHeader(http.StatusOK)
    w.Write([]byte("Login successful"))
}

func (as *AuthService) SessionAuthMiddleware(next http.HandlerFunc) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        cookie, err := r.Cookie("session_id")
        if err != nil {
            http.Error(w, "Session required", http.StatusUnauthorized)
            return
        }

        session, exists := as.sessionStore.GetSession(cookie.Value)
        if !exists {
            http.Error(w, "Invalid session", http.StatusUnauthorized)
            return
        }

        // Add user info to context
        ctx := context.WithValue(r.Context(), "session", session)
        next(w, r.WithContext(ctx))
    }
}
```

## 🔒 Security Best Practices

### 1. Password Security

```go
// Strong password requirements
func validatePassword(password string) error {
    if len(password) < 8 {
        return fmt.Errorf("password must be at least 8 characters")
    }

    hasUpper := false
    hasLower := false
    hasDigit := false
    hasSpecial := false

    for _, char := range password {
        switch {
        case 'A' <= char && char <= 'Z':
            hasUpper = true
        case 'a' <= char && char <= 'z':
            hasLower = true
        case '0' <= char && char <= '9':
            hasDigit = true
        case strings.ContainsRune("!@#$%^&*()_+-=[]{}|;:,.<>?", char):
            hasSpecial = true
        }
    }

    if !hasUpper || !hasLower || !hasDigit || !hasSpecial {
        return fmt.Errorf("password must contain uppercase, lowercase, digit, and special character")
    }

    return nil
}
```

### 2. Rate Limiting

```go
type RateLimiter struct {
    attempts map[string][]time.Time
    mutex    sync.RWMutex
    limit    int
    window   time.Duration
}

func (rl *RateLimiter) Allow(clientIP string) bool {
    rl.mutex.Lock()
    defer rl.mutex.Unlock()

    now := time.Now()
    cutoff := now.Add(-rl.window)

    // Clean old attempts
    if attempts, exists := rl.attempts[clientIP]; exists {
        var validAttempts []time.Time
        for _, attempt := range attempts {
            if attempt.After(cutoff) {
                validAttempts = append(validAttempts, attempt)
            }
        }
        rl.attempts[clientIP] = validAttempts
    }

    // Check limit
    if len(rl.attempts[clientIP]) >= rl.limit {
        return false
    }

    // Add current attempt
    rl.attempts[clientIP] = append(rl.attempts[clientIP], now)
    return true
}
```

### 3. CSRF Protection

```go
func CSRFMiddleware(next http.HandlerFunc) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        if r.Method == "POST" || r.Method == "PUT" || r.Method == "DELETE" {
            // Check CSRF token
            csrfToken := r.Header.Get("X-CSRF-Token")
            cookie, err := r.Cookie("csrf_token")

            if err != nil || csrfToken != cookie.Value {
                http.Error(w, "Invalid CSRF token", http.StatusForbidden)
                return
            }
        }

        next(w, r)
    }
}
```

## 🏢 Industry Insights

### Meta's Authentication

- **Facebook Login**: OAuth2 for third-party apps
- **Internal SSO**: SAML for enterprise integration
- **Mobile**: Biometric authentication
- **Security**: Multi-factor authentication required

### Google's Authentication

- **Google Sign-In**: OAuth2 + OpenID Connect
- **2FA**: TOTP and SMS verification
- **Advanced Protection**: Hardware security keys
- **Zero Trust**: Continuous authentication

### Amazon's Authentication

- **AWS IAM**: Role-based access control
- **MFA**: Hardware and software tokens
- **STS**: Temporary credentials
- **Cognito**: User pools and identity pools

## 🎯 Interview Questions

### Basic Level

1. **What's the difference between authentication and authorization?**

   - Authentication: Verifying identity
   - Authorization: Determining permissions

2. **Explain JWT structure?**

   - Header: Algorithm and token type
   - Payload: Claims and user data
   - Signature: Verification signature

3. **What are the advantages of JWT over sessions?**
   - Stateless: No server-side storage
   - Scalable: Works across multiple servers
   - Self-contained: All info in token

### Intermediate Level

4. **How do you implement secure password storage?**

   ```go
   // Use bcrypt with appropriate cost
   hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
   if err != nil {
       return err
   }
   ```

5. **Explain OAuth2 flow?**

   - Authorization request
   - User consent
   - Authorization code
   - Token exchange
   - Resource access

6. **How do you handle token refresh?**
   - Short-lived access tokens
   - Long-lived refresh tokens
   - Automatic token renewal
   - Secure token storage

### Advanced Level

7. **How do you implement multi-factor authentication?**

   ```go
   type MFAProvider interface {
       GenerateSecret(userID string) (string, error)
       ValidateCode(secret, code string) bool
   }

   type TOTPProvider struct {
       issuer string
   }

   func (t *TOTPProvider) GenerateSecret(userID string) (string, error) {
       key, err := totp.Generate(totp.GenerateOpts{
           Issuer:      t.issuer,
           AccountName: userID,
       })
       return key.Secret(), err
   }
   ```

8. **How do you implement single sign-on (SSO)?**

   - SAML for enterprise
   - OAuth2 + OpenID Connect
   - Centralized identity provider
   - Trust relationships

9. **How do you handle authentication in microservices?**
   - API gateway authentication
   - Service-to-service tokens
   - Distributed session management
   - Token validation service

---

**Next**: [Authorization](./Authorization.md) - RBAC, ABAC, and policy engines
