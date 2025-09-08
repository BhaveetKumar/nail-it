# 🛡️ Authorization: RBAC, ABAC, and Policy Engines

> **Master authorization patterns for secure access control in backend systems**

## 📚 Concept

Authorization determines what authenticated users can do within a system. It's the process of granting or denying access to resources based on user permissions.

### Authorization Models

1. **RBAC (Role-Based Access Control)**: Permissions based on user roles
2. **ABAC (Attribute-Based Access Control)**: Permissions based on attributes
3. **ACL (Access Control Lists)**: Direct user-resource permissions
4. **Policy-Based**: Rules engine for complex authorization

### Key Concepts

- **Subject**: User, service, or entity requesting access
- **Resource**: Object or data being accessed
- **Action**: Operation being performed (read, write, delete)
- **Context**: Environmental factors (time, location, device)

## 🏗️ Authorization Architecture

```
Request ──► Policy Engine ──► Decision ──► Resource Access
   │              │              │
   │              ▼              │
   │         Policy Rules        │
   │              │              │
   │              ▼              │
   └──► User Attributes ◄────────┘
```

## 🛠️ Hands-on Example

### RBAC Implementation (Go)

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "strings"
    "sync"
    "time"
)

type Permission string
type Role string
type Resource string
type Action string

const (
    // Actions
    ActionRead   Action = "read"
    ActionWrite  Action = "write"
    ActionDelete Action = "delete"
    ActionCreate Action = "create"

    // Resources
    ResourceUsers    Resource = "users"
    ResourcePosts    Resource = "posts"
    ResourceComments Resource = "comments"
    ResourceAdmin    Resource = "admin"

    // Roles
    RoleAdmin Role = "admin"
    RoleUser  Role = "user"
    RoleGuest Role = "guest"
)

type User struct {
    ID       int      `json:"id"`
    Username string   `json:"username"`
    Email    string   `json:"email"`
    Roles    []Role   `json:"roles"`
    CreatedAt time.Time `json:"created_at"`
}

type Permission struct {
    Resource Resource `json:"resource"`
    Action   Action   `json:"action"`
}

type RolePermissions struct {
    Role        Role        `json:"role"`
    Permissions []Permission `json:"permissions"`
}

type RBACService struct {
    users                map[int]User
    rolePermissions      map[Role][]Permission
    userRoles            map[int][]Role
    mutex                sync.RWMutex
}

func NewRBACService() *RBACService {
    rbac := &RBACService{
        users:           make(map[int]User),
        rolePermissions: make(map[Role][]Permission),
        userRoles:       make(map[int][]Role),
    }

    // Define role permissions
    rbac.rolePermissions[RoleAdmin] = []Permission{
        {Resource: ResourceUsers, Action: ActionRead},
        {Resource: ResourceUsers, Action: ActionWrite},
        {Resource: ResourceUsers, Action: ActionDelete},
        {Resource: ResourceUsers, Action: ActionCreate},
        {Resource: ResourcePosts, Action: ActionRead},
        {Resource: ResourcePosts, Action: ActionWrite},
        {Resource: ResourcePosts, Action: ActionDelete},
        {Resource: ResourcePosts, Action: ActionCreate},
        {Resource: ResourceComments, Action: ActionRead},
        {Resource: ResourceComments, Action: ActionWrite},
        {Resource: ResourceComments, Action: ActionDelete},
        {Resource: ResourceComments, Action: ActionCreate},
        {Resource: ResourceAdmin, Action: ActionRead},
        {Resource: ResourceAdmin, Action: ActionWrite},
    }

    rbac.rolePermissions[RoleUser] = []Permission{
        {Resource: ResourceUsers, Action: ActionRead},
        {Resource: ResourcePosts, Action: ActionRead},
        {Resource: ResourcePosts, Action: ActionWrite},
        {Resource: ResourcePosts, Action: ActionCreate},
        {Resource: ResourceComments, Action: ActionRead},
        {Resource: ResourceComments, Action: ActionWrite},
        {Resource: ResourceComments, Action: ActionCreate},
    }

    rbac.rolePermissions[RoleGuest] = []Permission{
        {Resource: ResourcePosts, Action: ActionRead},
        {Resource: ResourceComments, Action: ActionRead},
    }

    // Create sample users
    rbac.users[1] = User{
        ID:        1,
        Username:  "admin",
        Email:     "admin@example.com",
        Roles:     []Role{RoleAdmin},
        CreatedAt: time.Now(),
    }

    rbac.users[2] = User{
        ID:        2,
        Username:  "user1",
        Email:     "user1@example.com",
        Roles:     []Role{RoleUser},
        CreatedAt: time.Now(),
    }

    rbac.users[3] = User{
        ID:        3,
        Username:  "guest",
        Email:     "guest@example.com",
        Roles:     []Role{RoleGuest},
        CreatedAt: time.Now(),
    }

    // Set user roles
    rbac.userRoles[1] = []Role{RoleAdmin}
    rbac.userRoles[2] = []Role{RoleUser}
    rbac.userRoles[3] = []Role{RoleGuest}

    return rbac
}

func (rbac *RBACService) HasPermission(userID int, resource Resource, action Action) bool {
    rbac.mutex.RLock()
    defer rbac.mutex.RUnlock()

    userRoles, exists := rbac.userRoles[userID]
    if !exists {
        return false
    }

    for _, role := range userRoles {
        permissions, exists := rbac.rolePermissions[role]
        if !exists {
            continue
        }

        for _, permission := range permissions {
            if permission.Resource == resource && permission.Action == action {
                return true
            }
        }
    }

    return false
}

func (rbac *RBACService) GetUserPermissions(userID int) []Permission {
    rbac.mutex.RLock()
    defer rbac.mutex.RUnlock()

    userRoles, exists := rbac.userRoles[userID]
    if !exists {
        return []Permission{}
    }

    var permissions []Permission
    permissionMap := make(map[Permission]bool)

    for _, role := range userRoles {
        rolePermissions, exists := rbac.rolePermissions[role]
        if !exists {
            continue
        }

        for _, permission := range rolePermissions {
            if !permissionMap[permission] {
                permissions = append(permissions, permission)
                permissionMap[permission] = true
            }
        }
    }

    return permissions
}

func (rbac *RBACService) AssignRole(userID int, role Role) error {
    rbac.mutex.Lock()
    defer rbac.mutex.Unlock()

    if _, exists := rbac.users[userID]; !exists {
        return fmt.Errorf("user not found")
    }

    if _, exists := rbac.rolePermissions[role]; !exists {
        return fmt.Errorf("role not found")
    }

    // Check if user already has this role
    for _, existingRole := range rbac.userRoles[userID] {
        if existingRole == role {
            return fmt.Errorf("user already has this role")
        }
    }

    rbac.userRoles[userID] = append(rbac.userRoles[userID], role)
    return nil
}

func (rbac *RBACService) RemoveRole(userID int, role Role) error {
    rbac.mutex.Lock()
    defer rbac.mutex.Unlock()

    userRoles, exists := rbac.userRoles[userID]
    if !exists {
        return fmt.Errorf("user not found")
    }

    var newRoles []Role
    for _, existingRole := range userRoles {
        if existingRole != role {
            newRoles = append(newRoles, existingRole)
        }
    }

    rbac.userRoles[userID] = newRoles
    return nil
}

// Authorization middleware
func (rbac *RBACService) RequirePermission(resource Resource, action Action) func(http.HandlerFunc) http.HandlerFunc {
    return func(next http.HandlerFunc) http.HandlerFunc {
        return func(w http.ResponseWriter, r *http.Request) {
            // Get user ID from context (set by auth middleware)
            userID, ok := r.Context().Value("user_id").(int)
            if !ok {
                http.Error(w, "User not authenticated", http.StatusUnauthorized)
                return
            }

            if !rbac.HasPermission(userID, resource, action) {
                http.Error(w, "Insufficient permissions", http.StatusForbidden)
                return
            }

            next(w, r)
        }
    }
}

// API endpoints
func (rbac *RBACService) GetUsers(w http.ResponseWriter, r *http.Request) {
    users := make([]User, 0, len(rbac.users))
    for _, user := range rbac.users {
        users = append(users, user)
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]interface{}{
        "users": users,
        "total": len(users),
    })
}

func (rbac *RBACService) GetUser(w http.ResponseWriter, r *http.Request) {
    // Extract user ID from URL
    userID := extractUserIDFromURL(r.URL.Path)
    if userID == 0 {
        http.Error(w, "Invalid user ID", http.StatusBadRequest)
        return
    }

    user, exists := rbac.users[userID]
    if !exists {
        http.Error(w, "User not found", http.StatusNotFound)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

func (rbac *RBACService) CreateUser(w http.ResponseWriter, r *http.Request) {
    var newUser User
    if err := json.NewDecoder(r.Body).Decode(&newUser); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }

    // Generate new ID
    newUser.ID = len(rbac.users) + 1
    newUser.CreatedAt = time.Now()

    rbac.mutex.Lock()
    rbac.users[newUser.ID] = newUser
    rbac.userRoles[newUser.ID] = []Role{RoleUser} // Default role
    rbac.mutex.Unlock()

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusCreated)
    json.NewEncoder(w).Encode(newUser)
}

func (rbac *RBACService) DeleteUser(w http.ResponseWriter, r *http.Request) {
    userID := extractUserIDFromURL(r.URL.Path)
    if userID == 0 {
        http.Error(w, "Invalid user ID", http.StatusBadRequest)
        return
    }

    rbac.mutex.Lock()
    delete(rbac.users, userID)
    delete(rbac.userRoles, userID)
    rbac.mutex.Unlock()

    w.WriteHeader(http.StatusNoContent)
}

func (rbac *RBACService) GetUserPermissions(w http.ResponseWriter, r *http.Request) {
    userID := extractUserIDFromURL(r.URL.Path)
    if userID == 0 {
        http.Error(w, "Invalid user ID", http.StatusBadRequest)
        return
    }

    permissions := rbac.GetUserPermissions(userID)

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]interface{}{
        "user_id":     userID,
        "permissions": permissions,
    })
}

func (rbac *RBACService) AssignRoleToUser(w http.ResponseWriter, r *http.Request) {
    userID := extractUserIDFromURL(r.URL.Path)
    if userID == 0 {
        http.Error(w, "Invalid user ID", http.StatusBadRequest)
        return
    }

    var req struct {
        Role Role `json:"role"`
    }

    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }

    if err := rbac.AssignRole(userID, req.Role); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    w.WriteHeader(http.StatusOK)
    w.Write([]byte("Role assigned successfully"))
}

func extractUserIDFromURL(path string) int {
    // Simple extraction - in real app, use proper routing
    parts := strings.Split(path, "/")
    if len(parts) >= 3 {
        if parts[len(parts)-1] != "" {
            // Try to parse as int
            if id, err := strconv.Atoi(parts[len(parts)-1]); err == nil {
                return id
            }
        }
    }
    return 0
}

func main() {
    rbac := NewRBACService()

    // Public endpoints
    http.HandleFunc("/users", rbac.RequirePermission(ResourceUsers, ActionRead)(rbac.GetUsers))
    http.HandleFunc("/users/", func(w http.ResponseWriter, r *http.Request) {
        if strings.HasSuffix(r.URL.Path, "/permissions") {
            rbac.RequirePermission(ResourceUsers, ActionRead)(rbac.GetUserPermissions)(w, r)
        } else if strings.HasSuffix(r.URL.Path, "/assign-role") {
            rbac.RequirePermission(ResourceUsers, ActionWrite)(rbac.AssignRoleToUser)(w, r)
        } else {
            switch r.Method {
            case http.MethodGet:
                rbac.RequirePermission(ResourceUsers, ActionRead)(rbac.GetUser)(w, r)
            case http.MethodDelete:
                rbac.RequirePermission(ResourceUsers, ActionDelete)(rbac.DeleteUser)(w, r)
            default:
                http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
            }
        }
    })

    http.HandleFunc("/users", func(w http.ResponseWriter, r *http.Request) {
        if r.Method == http.MethodPost {
            rbac.RequirePermission(ResourceUsers, ActionCreate)(rbac.CreateUser)(w, r)
        } else {
            rbac.RequirePermission(ResourceUsers, ActionRead)(rbac.GetUsers)(w, r)
        }
    })

    log.Println("RBAC server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

### ABAC Implementation

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "strings"
    "time"
)

type Attribute struct {
    Name  string      `json:"name"`
    Value interface{} `json:"value"`
}

type Policy struct {
    ID          string     `json:"id"`
    Name        string     `json:"name"`
    Description string     `json:"description"`
    Rules       []Rule     `json:"rules"`
    Effect      string     `json:"effect"` // "allow" or "deny"
}

type Rule struct {
    Resource  string                 `json:"resource"`
    Action    string                 `json:"action"`
    Condition map[string]interface{} `json:"condition"`
}

type ABACService struct {
    policies map[string]Policy
    users    map[int]map[string]interface{}
    resources map[string]map[string]interface{}
}

func NewABACService() *ABACService {
    abac := &ABACService{
        policies:  make(map[string]Policy),
        users:     make(map[int]map[string]interface{}),
        resources: make(map[string]map[string]interface{}),
    }

    // Define sample policies
    abac.policies["user-read-own-data"] = Policy{
        ID:          "user-read-own-data",
        Name:        "Users can read their own data",
        Description: "Allow users to read their own profile data",
        Rules: []Rule{
            {
                Resource: "users",
                Action:   "read",
                Condition: map[string]interface{}{
                    "user_id": "resource.owner_id",
                },
            },
        },
        Effect: "allow",
    }

    abac.policies["admin-full-access"] = Policy{
        ID:          "admin-full-access",
        Name:        "Admin full access",
        Description: "Admins have full access to all resources",
        Rules: []Rule{
            {
                Resource: "*",
                Action:   "*",
                Condition: map[string]interface{}{
                    "role": "admin",
                },
            },
        },
        Effect: "allow",
    }

    abac.policies["business-hours-only"] = Policy{
        ID:          "business-hours-only",
        Name:        "Business hours access",
        Description: "Allow access only during business hours",
        Rules: []Rule{
            {
                Resource: "sensitive-data",
                Action:   "read",
                Condition: map[string]interface{}{
                    "time.hour": map[string]interface{}{
                        "gte": 9,
                        "lte": 17,
                    },
                },
            },
        },
        Effect: "allow",
    }

    // Sample user attributes
    abac.users[1] = map[string]interface{}{
        "id":       1,
        "username": "admin",
        "role":     "admin",
        "department": "IT",
        "location": "US",
    }

    abac.users[2] = map[string]interface{}{
        "id":       2,
        "username": "user1",
        "role":     "user",
        "department": "Sales",
        "location": "US",
    }

    // Sample resource attributes
    abac.resources["users/1"] = map[string]interface{}{
        "owner_id": 1,
        "type":     "user",
        "sensitivity": "public",
    }

    abac.resources["users/2"] = map[string]interface{}{
        "owner_id": 2,
        "type":     "user",
        "sensitivity": "private",
    }

    return abac
}

func (abac *ABACService) EvaluatePolicy(userID int, resource string, action string, context map[string]interface{}) bool {
    userAttrs := abac.users[userID]
    resourceAttrs := abac.resources[resource]

    // Add current time to context
    now := time.Now()
    context["time"] = map[string]interface{}{
        "hour": now.Hour(),
        "day":  now.Weekday().String(),
    }

    // Evaluate each policy
    for _, policy := range abac.policies {
        if abac.evaluatePolicyRules(policy, userAttrs, resourceAttrs, context, resource, action) {
            return policy.Effect == "allow"
        }
    }

    // Default deny
    return false
}

func (abac *ABACService) evaluatePolicyRules(policy Policy, userAttrs, resourceAttrs map[string]interface{}, context map[string]interface{}, resource, action string) bool {
    for _, rule := range policy.Rules {
        // Check resource and action match
        if rule.Resource != "*" && rule.Resource != resource {
            continue
        }
        if rule.Action != "*" && rule.Action != action {
            continue
        }

        // Evaluate conditions
        if abac.evaluateConditions(rule.Condition, userAttrs, resourceAttrs, context) {
            return true
        }
    }

    return false
}

func (abac *ABACService) evaluateConditions(conditions map[string]interface{}, userAttrs, resourceAttrs map[string]interface{}, context map[string]interface{}) bool {
    for key, value := range conditions {
        var actualValue interface{}

        // Resolve attribute references
        if strings.HasPrefix(key, "user.") {
            attrName := strings.TrimPrefix(key, "user.")
            actualValue = userAttrs[attrName]
        } else if strings.HasPrefix(key, "resource.") {
            attrName := strings.TrimPrefix(key, "resource.")
            actualValue = resourceAttrs[attrName]
        } else if strings.HasPrefix(key, "context.") {
            attrName := strings.TrimPrefix(key, "context.")
            actualValue = context[attrName]
        } else {
            actualValue = context[key]
        }

        // Evaluate condition
        if !abac.evaluateCondition(actualValue, value) {
            return false
        }
    }

    return true
}

func (abac *ABACService) evaluateCondition(actual, expected interface{}) bool {
    switch expected := expected.(type) {
    case map[string]interface{}:
        // Range conditions (e.g., {"gte": 9, "lte": 17})
        if gte, ok := expected["gte"]; ok {
            if !abac.compareValues(actual, gte, "gte") {
                return false
            }
        }
        if lte, ok := expected["lte"]; ok {
            if !abac.compareValues(actual, lte, "lte") {
                return false
            }
        }
        if gt, ok := expected["gt"]; ok {
            if !abac.compareValues(actual, gt, "gt") {
                return false
            }
        }
        if lt, ok := expected["lt"]; ok {
            if !abac.compareValues(actual, lt, "lt") {
                return false
            }
        }
        return true
    default:
        // Direct comparison
        return actual == expected
    }
}

func (abac *ABACService) compareValues(actual, expected interface{}, operator string) bool {
    // Simple comparison - in real implementation, handle different types
    actualFloat, ok1 := actual.(float64)
    expectedFloat, ok2 := expected.(float64)

    if !ok1 || !ok2 {
        return false
    }

    switch operator {
    case "gte":
        return actualFloat >= expectedFloat
    case "lte":
        return actualFloat <= expectedFloat
    case "gt":
        return actualFloat > expectedFloat
    case "lt":
        return actualFloat < expectedFloat
    default:
        return false
    }
}

// ABAC middleware
func (abac *ABACService) RequireABACPermission(resource string, action string) func(http.HandlerFunc) http.HandlerFunc {
    return func(next http.HandlerFunc) http.HandlerFunc {
        return func(w http.ResponseWriter, r *http.Request) {
            userID, ok := r.Context().Value("user_id").(int)
            if !ok {
                http.Error(w, "User not authenticated", http.StatusUnauthorized)
                return
            }

            // Build context from request
            context := map[string]interface{}{
                "ip":        r.RemoteAddr,
                "user_agent": r.UserAgent(),
                "method":    r.Method,
            }

            if !abac.EvaluatePolicy(userID, resource, action, context) {
                http.Error(w, "Access denied", http.StatusForbidden)
                return
            }

            next(w, r)
        }
    }
}
```

### Policy Engine Implementation

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "regexp"
    "strings"
    "time"
)

type PolicyEngine struct {
    policies []Policy
    rules    []Rule
}

type Rule struct {
    ID          string                 `json:"id"`
    Name        string                 `json:"name"`
    Description string                 `json:"description"`
    Conditions  []Condition            `json:"conditions"`
    Actions     []string               `json:"actions"`
    Effect      string                 `json:"effect"`
    Priority    int                    `json:"priority"`
    Metadata    map[string]interface{} `json:"metadata"`
}

type Condition struct {
    Attribute string      `json:"attribute"`
    Operator  string      `json:"operator"`
    Value     interface{} `json:"value"`
}

type Policy struct {
    ID          string                 `json:"id"`
    Name        string                 `json:"name"`
    Description string                 `json:"description"`
    Rules       []string               `json:"rules"` // Rule IDs
    Effect      string                 `json:"effect"`
    Priority    int                    `json:"priority"`
    Metadata    map[string]interface{} `json:"metadata"`
}

func NewPolicyEngine() *PolicyEngine {
    engine := &PolicyEngine{
        policies: make([]Policy, 0),
        rules:    make([]Rule, 0),
    }

    // Define sample rules
    engine.rules = []Rule{
        {
            ID:          "rule-1",
            Name:        "Admin full access",
            Description: "Admins have full access",
            Conditions: []Condition{
                {Attribute: "user.role", Operator: "equals", Value: "admin"},
            },
            Actions:  []string{"*"},
            Effect:   "allow",
            Priority: 100,
        },
        {
            ID:          "rule-2",
            Name:        "User read own data",
            Description: "Users can read their own data",
            Conditions: []Condition{
                {Attribute: "user.id", Operator: "equals", Value: "resource.owner_id"},
                {Attribute: "action", Operator: "equals", Value: "read"},
            },
            Actions:  []string{"read"},
            Effect:   "allow",
            Priority: 50,
        },
        {
            ID:          "rule-3",
            Name:        "Business hours restriction",
            Description: "Restrict access during non-business hours",
            Conditions: []Condition{
                {Attribute: "time.hour", Operator: "lt", Value: 9},
                {Attribute: "resource.sensitivity", Operator: "equals", Value: "high"},
            },
            Actions:  []string{"*"},
            Effect:   "deny",
            Priority: 75,
        },
        {
            ID:          "rule-4",
            Name:        "Rate limiting",
            Description: "Limit requests per user",
            Conditions: []Condition{
                {Attribute: "user.requests_per_minute", Operator: "gt", Value: 100},
            },
            Actions:  []string{"*"},
            Effect:   "deny",
            Priority: 90,
        },
    }

    // Define sample policies
    engine.policies = []Policy{
        {
            ID:          "policy-1",
            Name:        "User access policy",
            Description: "Policy for regular users",
            Rules:       []string{"rule-2", "rule-3"},
            Effect:      "allow",
            Priority:    50,
        },
        {
            ID:          "policy-2",
            Name:        "Admin access policy",
            Description: "Policy for administrators",
            Rules:       []string{"rule-1"},
            Effect:      "allow",
            Priority:    100,
        },
        {
            ID:          "policy-3",
            Name:        "Rate limiting policy",
            Description: "Global rate limiting",
            Rules:       []string{"rule-4"},
            Effect:      "deny",
            Priority:    90,
        },
    }

    return engine
}

func (pe *PolicyEngine) Evaluate(userAttrs, resourceAttrs, context map[string]interface{}, action string) (bool, string, error) {
    // Sort policies by priority (highest first)
    policies := make([]Policy, len(pe.policies))
    copy(policies, pe.policies)

    for i := 0; i < len(policies); i++ {
        for j := i + 1; j < len(policies); j++ {
            if policies[i].Priority < policies[j].Priority {
                policies[i], policies[j] = policies[j], policies[i]
            }
        }
    }

    // Evaluate each policy
    for _, policy := range policies {
        if pe.evaluatePolicy(policy, userAttrs, resourceAttrs, context, action) {
            return policy.Effect == "allow", policy.ID, nil
        }
    }

    // Default deny
    return false, "default-deny", nil
}

func (pe *PolicyEngine) evaluatePolicy(policy Policy, userAttrs, resourceAttrs, context map[string]interface{}, action string) bool {
    // Check if any rule in the policy matches
    for _, ruleID := range policy.Rules {
        rule := pe.getRuleByID(ruleID)
        if rule == nil {
            continue
        }

        if pe.evaluateRule(*rule, userAttrs, resourceAttrs, context, action) {
            return true
        }
    }

    return false
}

func (pe *PolicyEngine) evaluateRule(rule Rule, userAttrs, resourceAttrs, context map[string]interface{}, action string) bool {
    // Check if action matches
    if !pe.actionMatches(rule.Actions, action) {
        return false
    }

    // Evaluate all conditions
    for _, condition := range rule.Conditions {
        if !pe.evaluateCondition(condition, userAttrs, resourceAttrs, context) {
            return false
        }
    }

    return true
}

func (pe *PolicyEngine) actionMatches(ruleActions []string, action string) bool {
    for _, ruleAction := range ruleActions {
        if ruleAction == "*" || ruleAction == action {
            return true
        }
    }
    return false
}

func (pe *PolicyEngine) evaluateCondition(condition Condition, userAttrs, resourceAttrs, context map[string]interface{}) bool {
    // Resolve attribute value
    actualValue := pe.resolveAttribute(condition.Attribute, userAttrs, resourceAttrs, context)

    // Evaluate condition based on operator
    switch condition.Operator {
    case "equals":
        return actualValue == condition.Value
    case "not_equals":
        return actualValue != condition.Value
    case "gt":
        return pe.compareValues(actualValue, condition.Value, "gt")
    case "gte":
        return pe.compareValues(actualValue, condition.Value, "gte")
    case "lt":
        return pe.compareValues(actualValue, condition.Value, "lt")
    case "lte":
        return pe.compareValues(actualValue, condition.Value, "lte")
    case "in":
        return pe.valueInList(actualValue, condition.Value)
    case "not_in":
        return !pe.valueInList(actualValue, condition.Value)
    case "regex":
        return pe.regexMatch(actualValue, condition.Value)
    case "exists":
        return actualValue != nil
    case "not_exists":
        return actualValue == nil
    default:
        return false
    }
}

func (pe *PolicyEngine) resolveAttribute(attribute string, userAttrs, resourceAttrs, context map[string]interface{}) interface{} {
    if strings.HasPrefix(attribute, "user.") {
        attrName := strings.TrimPrefix(attribute, "user.")
        return userAttrs[attrName]
    } else if strings.HasPrefix(attribute, "resource.") {
        attrName := strings.TrimPrefix(attribute, "resource.")
        return resourceAttrs[attrName]
    } else if strings.HasPrefix(attribute, "context.") {
        attrName := strings.TrimPrefix(attribute, "context.")
        return context[attrName]
    } else {
        return context[attribute]
    }
}

func (pe *PolicyEngine) compareValues(actual, expected interface{}, operator string) bool {
    actualFloat, ok1 := actual.(float64)
    expectedFloat, ok2 := expected.(float64)

    if !ok1 || !ok2 {
        return false
    }

    switch operator {
    case "gt":
        return actualFloat > expectedFloat
    case "gte":
        return actualFloat >= expectedFloat
    case "lt":
        return actualFloat < expectedFloat
    case "lte":
        return actualFloat <= expectedFloat
    default:
        return false
    }
}

func (pe *PolicyEngine) valueInList(value, list interface{}) bool {
    listSlice, ok := list.([]interface{})
    if !ok {
        return false
    }

    for _, item := range listSlice {
        if item == value {
            return true
        }
    }
    return false
}

func (pe *PolicyEngine) regexMatch(value, pattern interface{}) bool {
    valueStr, ok1 := value.(string)
    patternStr, ok2 := pattern.(string)

    if !ok1 || !ok2 {
        return false
    }

    matched, err := regexp.MatchString(patternStr, valueStr)
    return err == nil && matched
}

func (pe *PolicyEngine) getRuleByID(ruleID string) *Rule {
    for _, rule := range pe.rules {
        if rule.ID == ruleID {
            return &rule
        }
    }
    return nil
}

// API endpoints for policy management
func (pe *PolicyEngine) GetPolicies(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]interface{}{
        "policies": pe.policies,
        "total":    len(pe.policies),
    })
}

func (pe *PolicyEngine) GetRules(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]interface{}{
        "rules": pe.rules,
        "total": len(pe.rules),
    })
}

func (pe *PolicyEngine) EvaluateRequest(w http.ResponseWriter, r *http.Request) {
    var req struct {
        UserAttrs     map[string]interface{} `json:"user_attrs"`
        ResourceAttrs map[string]interface{} `json:"resource_attrs"`
        Context       map[string]interface{} `json:"context"`
        Action        string                 `json:"action"`
    }

    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }

    // Add current time to context
    now := time.Now()
    req.Context["time"] = map[string]interface{}{
        "hour": now.Hour(),
        "day":  now.Weekday().String(),
    }

    allowed, policyID, err := pe.Evaluate(req.UserAttrs, req.ResourceAttrs, req.Context, req.Action)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    response := map[string]interface{}{
        "allowed":   allowed,
        "policy_id": policyID,
        "decision":  "allow",
    }

    if !allowed {
        response["decision"] = "deny"
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}
```

## 🚀 Best Practices

### 1. Principle of Least Privilege

```go
// Grant minimum required permissions
func (rbac *RBACService) AssignMinimalPermissions(userID int, requiredPermissions []Permission) error {
    // Remove all existing permissions
    rbac.userRoles[userID] = []Role{}

    // Assign only required permissions
    for _, permission := range requiredPermissions {
        // Find role with minimal permissions that includes this permission
        role := rbac.findMinimalRole(permission)
        if role != "" {
            rbac.AssignRole(userID, role)
        }
    }

    return nil
}
```

### 2. Permission Inheritance

```go
type RoleHierarchy struct {
    roles map[Role][]Role // Parent -> Children
}

func (rh *RoleHierarchy) GetInheritedRoles(role Role) []Role {
    var allRoles []Role
    visited := make(map[Role]bool)

    var dfs func(Role)
    dfs = func(current Role) {
        if visited[current] {
            return
        }
        visited[current] = true
        allRoles = append(allRoles, current)

        for _, child := range rh.roles[current] {
            dfs(child)
        }
    }

    dfs(role)
    return allRoles
}
```

### 3. Dynamic Permission Evaluation

```go
func (abac *ABACService) EvaluateWithContext(userID int, resource string, action string, requestContext map[string]interface{}) bool {
    // Add request-specific context
    requestContext["timestamp"] = time.Now()
    requestContext["ip"] = requestContext["ip"]
    requestContext["user_agent"] = requestContext["user_agent"]

    // Evaluate with enriched context
    return abac.EvaluatePolicy(userID, resource, action, requestContext)
}
```

## 🏢 Industry Insights

### Google's Authorization

- **BeyondCorp**: Zero-trust network access
- **IAM**: Fine-grained access control
- **Context-aware**: Device, location, time-based policies
- **Risk-based**: Adaptive authentication

### Amazon's Authorization

- **IAM Policies**: JSON-based policy language
- **Resource-based**: Attach policies to resources
- **Identity-based**: Attach policies to users/roles
- **Cross-account**: Assume roles across accounts

### Microsoft's Authorization

- **Azure RBAC**: Role-based access control
- **Conditional Access**: Context-aware policies
- **Privileged Identity Management**: Just-in-time access
- **Access Reviews**: Regular permission audits

## 🎯 Interview Questions

### Basic Level

1. **What's the difference between RBAC and ABAC?**

   - RBAC: Permissions based on roles
   - ABAC: Permissions based on attributes and context

2. **Explain the principle of least privilege?**

   - Grant minimum required permissions
   - Regular permission audits
   - Temporary access when possible

3. **What are the components of an authorization system?**
   - Subjects (users, services)
   - Resources (data, services)
   - Actions (read, write, delete)
   - Policies (rules, conditions)

### Intermediate Level

4. **How do you implement role inheritance?**

   ```go
   type RoleHierarchy struct {
       roles map[Role][]Role
   }

   func (rh *RoleHierarchy) GetInheritedPermissions(role Role) []Permission {
       var permissions []Permission
       for _, inheritedRole := range rh.GetInheritedRoles(role) {
           permissions = append(permissions, rh.getRolePermissions(inheritedRole)...)
       }
       return permissions
   }
   ```

5. **How do you handle permission caching?**

   - Cache user permissions in memory
   - Invalidate on role changes
   - Use Redis for distributed caching
   - Implement cache warming strategies

6. **Explain attribute-based access control?**
   - User attributes (role, department, location)
   - Resource attributes (sensitivity, owner, type)
   - Context attributes (time, IP, device)
   - Policy evaluation engine

### Advanced Level

7. **How do you implement fine-grained permissions?**

   ```go
   type FineGrainedPermission struct {
       Resource    string                 `json:"resource"`
       Action      string                 `json:"action"`
       Conditions  map[string]interface{} `json:"conditions"`
       Attributes  map[string]interface{} `json:"attributes"`
   }

   func (fgp *FineGrainedPermission) Evaluate(userAttrs, context map[string]interface{}) bool {
       // Evaluate complex conditions
       for attr, condition := range fgp.Conditions {
           if !evaluateCondition(userAttrs[attr], condition) {
               return false
           }
       }
       return true
   }
   ```

8. **How do you implement policy versioning?**

   - Version policies with timestamps
   - A/B testing for policy changes
   - Rollback capabilities
   - Audit trail for policy changes

9. **How do you handle authorization in microservices?**
   - Centralized policy engine
   - Service-specific policies
   - Cross-service authorization
   - Distributed policy evaluation

---

**Next**: [Caching Strategies](./CachingStrategies.md) - Redis, Memcached, CDN, and cache patterns
