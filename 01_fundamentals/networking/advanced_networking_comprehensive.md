# Advanced Networking Comprehensive

Comprehensive guide to advanced networking for senior backend engineers.

## 🎯 Advanced Network Protocols

### Custom Protocol Implementation
```go
// Advanced Custom Protocol Implementation
package networking

import (
    "bytes"
    "encoding/binary"
    "fmt"
    "io"
    "net"
    "sync"
    "time"
)

type CustomProtocol struct {
    conn        net.Conn
    buffer      []byte
    mutex       sync.RWMutex
    timeout     time.Duration
    maxMsgSize  int
    compression bool
    encryption  bool
}

type Message struct {
    Header  MessageHeader `json:"header"`
    Payload []byte        `json:"payload"`
}

type MessageHeader struct {
    Version     uint8     `json:"version"`
    Type        uint8     `json:"type"`
    Length      uint32    `json:"length"`
    Sequence    uint32    `json:"sequence"`
    Timestamp   uint64    `json:"timestamp"`
    Checksum    uint32    `json:"checksum"`
    Flags       uint16    `json:"flags"`
    Reserved    [4]byte   `json:"reserved"`
}

const (
    MessageTypeRequest  = 0x01
    MessageTypeResponse = 0x02
    MessageTypePing     = 0x03
    MessageTypePong     = 0x04
    MessageTypeError    = 0x05
)

const (
    FlagCompressed = 0x01
    FlagEncrypted  = 0x02
    FlagReliable   = 0x04
    FlagUrgent     = 0x08
)

func NewCustomProtocol(conn net.Conn, timeout time.Duration, maxMsgSize int) *CustomProtocol {
    return &CustomProtocol{
        conn:       conn,
        buffer:     make([]byte, maxMsgSize),
        timeout:    timeout,
        maxMsgSize: maxMsgSize,
    }
}

func (cp *CustomProtocol) SendMessage(msg *Message) error {
    cp.mutex.Lock()
    defer cp.mutex.Unlock()
    
    // Set timestamp
    msg.Header.Timestamp = uint64(time.Now().UnixNano())
    
    // Calculate checksum
    msg.Header.Checksum = cp.calculateChecksum(msg)
    
    // Serialize message
    data, err := cp.serializeMessage(msg)
    if err != nil {
        return fmt.Errorf("failed to serialize message: %w", err)
    }
    
    // Set connection timeout
    if err := cp.conn.SetWriteDeadline(time.Now().Add(cp.timeout)); err != nil {
        return fmt.Errorf("failed to set write deadline: %w", err)
    }
    
    // Send message
    if _, err := cp.conn.Write(data); err != nil {
        return fmt.Errorf("failed to write message: %w", err)
    }
    
    return nil
}

func (cp *CustomProtocol) ReceiveMessage() (*Message, error) {
    cp.mutex.Lock()
    defer cp.mutex.Unlock()
    
    // Set connection timeout
    if err := cp.conn.SetReadDeadline(time.Now().Add(cp.timeout)); err != nil {
        return nil, fmt.Errorf("failed to set read deadline: %w", err)
    }
    
    // Read header first
    headerSize := binary.Size(MessageHeader{})
    if _, err := io.ReadFull(cp.conn, cp.buffer[:headerSize]); err != nil {
        return nil, fmt.Errorf("failed to read header: %w", err)
    }
    
    // Parse header
    header, err := cp.parseHeader(cp.buffer[:headerSize])
    if err != nil {
        return nil, fmt.Errorf("failed to parse header: %w", err)
    }
    
    // Validate message size
    if header.Length > uint32(cp.maxMsgSize) {
        return nil, fmt.Errorf("message too large: %d bytes", header.Length)
    }
    
    // Read payload
    payloadSize := int(header.Length) - headerSize
    if payloadSize > 0 {
        if _, err := io.ReadFull(cp.conn, cp.buffer[headerSize:headerSize+payloadSize]); err != nil {
            return nil, fmt.Errorf("failed to read payload: %w", err)
        }
    }
    
    // Create message
    msg := &Message{
        Header:  *header,
        Payload: cp.buffer[headerSize : headerSize+payloadSize],
    }
    
    // Verify checksum
    if !cp.verifyChecksum(msg) {
        return nil, fmt.Errorf("checksum verification failed")
    }
    
    return msg, nil
}

func (cp *CustomProtocol) serializeMessage(msg *Message) ([]byte, error) {
    var buf bytes.Buffer
    
    // Write header
    if err := binary.Write(&buf, binary.BigEndian, msg.Header); err != nil {
        return nil, err
    }
    
    // Write payload
    if _, err := buf.Write(msg.Payload); err != nil {
        return nil, err
    }
    
    return buf.Bytes(), nil
}

func (cp *CustomProtocol) parseHeader(data []byte) (*MessageHeader, error) {
    var header MessageHeader
    
    if err := binary.Read(bytes.NewReader(data), binary.BigEndian, &header); err != nil {
        return nil, err
    }
    
    return &header, nil
}

func (cp *CustomProtocol) calculateChecksum(msg *Message) uint32 {
    var checksum uint32
    
    // Calculate checksum for header (excluding checksum field)
    headerData, _ := cp.serializeHeaderWithoutChecksum(&msg.Header)
    for _, b := range headerData {
        checksum += uint32(b)
    }
    
    // Calculate checksum for payload
    for _, b := range msg.Payload {
        checksum += uint32(b)
    }
    
    return checksum
}

func (cp *CustomProtocol) serializeHeaderWithoutChecksum(header *MessageHeader) ([]byte, error) {
    var buf bytes.Buffer
    
    // Write all fields except checksum
    if err := binary.Write(&buf, binary.BigEndian, header.Version); err != nil {
        return nil, err
    }
    if err := binary.Write(&buf, binary.BigEndian, header.Type); err != nil {
        return nil, err
    }
    if err := binary.Write(&buf, binary.BigEndian, header.Length); err != nil {
        return nil, err
    }
    if err := binary.Write(&buf, binary.BigEndian, header.Sequence); err != nil {
        return nil, err
    }
    if err := binary.Write(&buf, binary.BigEndian, header.Timestamp); err != nil {
        return nil, err
    }
    if err := binary.Write(&buf, binary.BigEndian, header.Flags); err != nil {
        return nil, err
    }
    if err := binary.Write(&buf, binary.BigEndian, header.Reserved); err != nil {
        return nil, err
    }
    
    return buf.Bytes(), nil
}

func (cp *CustomProtocol) verifyChecksum(msg *Message) bool {
    expectedChecksum := cp.calculateChecksum(msg)
    return msg.Header.Checksum == expectedChecksum
}

// Connection Pool Implementation
type ConnectionPool struct {
    connections chan net.Conn
    factory     func() (net.Conn, error)
    maxSize     int
    timeout     time.Duration
    mutex       sync.RWMutex
    closed      bool
}

func NewConnectionPool(factory func() (net.Conn, error), maxSize int, timeout time.Duration) *ConnectionPool {
    return &ConnectionPool{
        connections: make(chan net.Conn, maxSize),
        factory:     factory,
        maxSize:     maxSize,
        timeout:     timeout,
    }
}

func (cp *ConnectionPool) Get() (net.Conn, error) {
    cp.mutex.RLock()
    if cp.closed {
        cp.mutex.RUnlock()
        return nil, fmt.Errorf("connection pool is closed")
    }
    cp.mutex.RUnlock()
    
    select {
    case conn := <-cp.connections:
        // Check if connection is still alive
        if cp.isConnectionAlive(conn) {
            return conn, nil
        }
        // Connection is dead, create a new one
        return cp.factory()
    case <-time.After(cp.timeout):
        return nil, fmt.Errorf("timeout waiting for connection")
    }
}

func (cp *ConnectionPool) Put(conn net.Conn) {
    cp.mutex.RLock()
    if cp.closed {
        cp.mutex.RUnlock()
        conn.Close()
        return
    }
    cp.mutex.RUnlock()
    
    select {
    case cp.connections <- conn:
        // Connection returned to pool
    default:
        // Pool is full, close the connection
        conn.Close()
    }
}

func (cp *ConnectionPool) Close() {
    cp.mutex.Lock()
    defer cp.mutex.Unlock()
    
    if cp.closed {
        return
    }
    
    cp.closed = true
    close(cp.connections)
    
    // Close all connections in the pool
    for {
        select {
        case conn := <-cp.connections:
            conn.Close()
        default:
            return
        }
    }
}

func (cp *ConnectionPool) isConnectionAlive(conn net.Conn) bool {
    // Set a short timeout for the check
    conn.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
    
    // Try to read one byte
    one := make([]byte, 1)
    _, err := conn.Read(one)
    
    // Reset the deadline
    conn.SetReadDeadline(time.Time{})
    
    return err == nil
}
```

### Advanced Load Balancing
```go
// Advanced Load Balancer with Multiple Algorithms
package networking

import (
    "context"
    "fmt"
    "math/rand"
    "sync"
    "time"
)

type LoadBalancer struct {
    servers      []*Server
    algorithm    LoadBalanceAlgorithm
    healthChecker *HealthChecker
    mutex        sync.RWMutex
}

type Server struct {
    ID          string    `json:"id"`
    Address     string    `json:"address"`
    Weight      int       `json:"weight"`
    Health      HealthStatus `json:"health"`
    Connections int       `json:"connections"`
    ResponseTime time.Duration `json:"response_time"`
    LastCheck   time.Time `json:"last_check"`
    mutex       sync.RWMutex
}

type HealthStatus int

const (
    Healthy HealthStatus = iota
    Unhealthy
    Unknown
)

type LoadBalanceAlgorithm interface {
    SelectServer(servers []*Server) *Server
    Name() string
}

// Round Robin Algorithm
type RoundRobinAlgorithm struct {
    current int
    mutex   sync.Mutex
}

func (rra *RoundRobinAlgorithm) SelectServer(servers []*Server) *Server {
    rra.mutex.Lock()
    defer rra.mutex.Unlock()
    
    healthyServers := rra.getHealthyServers(servers)
    if len(healthyServers) == 0 {
        return nil
    }
    
    server := healthyServers[rra.current%len(healthyServers)]
    rra.current++
    return server
}

func (rra *RoundRobinAlgorithm) Name() string {
    return "round_robin"
}

func (rra *RoundRobinAlgorithm) getHealthyServers(servers []*Server) []*Server {
    var healthy []*Server
    for _, server := range servers {
        if server.Health == Healthy {
            healthy = append(healthy, server)
        }
    }
    return healthy
}

// Weighted Round Robin Algorithm
type WeightedRoundRobinAlgorithm struct {
    current int
    mutex   sync.Mutex
}

func (wrra *WeightedRoundRobinAlgorithm) SelectServer(servers []*Server) *Server {
    wrra.mutex.Lock()
    defer wrra.mutex.Unlock()
    
    healthyServers := wrra.getHealthyServers(servers)
    if len(healthyServers) == 0 {
        return nil
    }
    
    totalWeight := wrra.calculateTotalWeight(healthyServers)
    if totalWeight == 0 {
        return healthyServers[0]
    }
    
    currentWeight := wrra.current % totalWeight
    for _, server := range healthyServers {
        currentWeight -= server.Weight
        if currentWeight < 0 {
            wrra.current++
            return server
        }
    }
    
    return healthyServers[0]
}

func (wrra *WeightedRoundRobinAlgorithm) Name() string {
    return "weighted_round_robin"
}

func (wrra *WeightedRoundRobinAlgorithm) getHealthyServers(servers []*Server) []*Server {
    var healthy []*Server
    for _, server := range servers {
        if server.Health == Healthy {
            healthy = append(healthy, server)
        }
    }
    return healthy
}

func (wrra *WeightedRoundRobinAlgorithm) calculateTotalWeight(servers []*Server) int {
    totalWeight := 0
    for _, server := range servers {
        totalWeight += server.Weight
    }
    return totalWeight
}

// Least Connections Algorithm
type LeastConnectionsAlgorithm struct{}

func (lca *LeastConnectionsAlgorithm) SelectServer(servers []*Server) *Server {
    healthyServers := lca.getHealthyServers(servers)
    if len(healthyServers) == 0 {
        return nil
    }
    
    var bestServer *Server
    minConnections := int(^uint(0) >> 1) // Max int
    
    for _, server := range healthyServers {
        server.mutex.RLock()
        connections := server.Connections
        server.mutex.RUnlock()
        
        if connections < minConnections {
            minConnections = connections
            bestServer = server
        }
    }
    
    return bestServer
}

func (lca *LeastConnectionsAlgorithm) Name() string {
    return "least_connections"
}

func (lca *LeastConnectionsAlgorithm) getHealthyServers(servers []*Server) []*Server {
    var healthy []*Server
    for _, server := range servers {
        if server.Health == Healthy {
            healthy = append(healthy, server)
        }
    }
    return healthy
}

// Least Response Time Algorithm
type LeastResponseTimeAlgorithm struct{}

func (lrta *LeastResponseTimeAlgorithm) SelectServer(servers []*Server) *Server {
    healthyServers := lrta.getHealthyServers(servers)
    if len(healthyServers) == 0 {
        return nil
    }
    
    var bestServer *Server
    minResponseTime := time.Duration(^uint64(0)) // Max duration
    
    for _, server := range healthyServers {
        server.mutex.RLock()
        responseTime := server.ResponseTime
        server.mutex.RUnlock()
        
        if responseTime < minResponseTime {
            minResponseTime = responseTime
            bestServer = server
        }
    }
    
    return bestServer
}

func (lrta *LeastResponseTimeAlgorithm) Name() string {
    return "least_response_time"
}

func (lrta *LeastResponseTimeAlgorithm) getHealthyServers(servers []*Server) []*Server {
    var healthy []*Server
    for _, server := range servers {
        if server.Health == Healthy {
            healthy = append(healthy, server)
        }
    }
    return healthy
}

// Random Algorithm
type RandomAlgorithm struct{}

func (ra *RandomAlgorithm) SelectServer(servers []*Server) *Server {
    healthyServers := ra.getHealthyServers(servers)
    if len(healthyServers) == 0 {
        return nil
    }
    
    index := rand.Intn(len(healthyServers))
    return healthyServers[index]
}

func (ra *RandomAlgorithm) Name() string {
    return "random"
}

func (ra *RandomAlgorithm) getHealthyServers(servers []*Server) []*Server {
    var healthy []*Server
    for _, server := range servers {
        if server.Health == Healthy {
            healthy = append(healthy, server)
        }
    }
    return healthy
}

// Consistent Hash Algorithm
type ConsistentHashAlgorithm struct {
    hashRing *ConsistentHash
    mutex    sync.RWMutex
}

func NewConsistentHashAlgorithm() *ConsistentHashAlgorithm {
    return &ConsistentHashAlgorithm{
        hashRing: NewConsistentHash(),
    }
}

func (cha *ConsistentHashAlgorithm) SelectServer(servers []*Server) *Server {
    cha.mutex.RLock()
    defer cha.mutex.RUnlock()
    
    healthyServers := cha.getHealthyServers(servers)
    if len(healthyServers) == 0 {
        return nil
    }
    
    // Use client IP or some other identifier for consistent hashing
    // For simplicity, we'll use a random key
    key := fmt.Sprintf("%d", rand.Intn(1000000))
    
    serverID := cha.hashRing.GetNode(key)
    for _, server := range healthyServers {
        if server.ID == serverID {
            return server
        }
    }
    
    return healthyServers[0]
}

func (cha *ConsistentHashAlgorithm) Name() string {
    return "consistent_hash"
}

func (cha *ConsistentHashAlgorithm) getHealthyServers(servers []*Server) []*Server {
    var healthy []*Server
    for _, server := range servers {
        if server.Health == Healthy {
            healthy = append(healthy, server)
        }
    }
    return healthy
}

func (cha *ConsistentHashAlgorithm) UpdateServers(servers []*Server) {
    cha.mutex.Lock()
    defer cha.mutex.Unlock()
    
    // Clear existing nodes
    cha.hashRing = NewConsistentHash()
    
    // Add healthy servers
    for _, server := range servers {
        if server.Health == Healthy {
            cha.hashRing.AddNode(server.ID)
        }
    }
}

// Load Balancer Implementation
func NewLoadBalancer(algorithm LoadBalanceAlgorithm) *LoadBalancer {
    return &LoadBalancer{
        algorithm:    algorithm,
        healthChecker: NewHealthChecker(),
    }
}

func (lb *LoadBalancer) AddServer(server *Server) {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()
    
    lb.servers = append(lb.servers, server)
    lb.healthChecker.AddServer(server)
}

func (lb *LoadBalancer) RemoveServer(serverID string) {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()
    
    for i, server := range lb.servers {
        if server.ID == serverID {
            lb.servers = append(lb.servers[:i], lb.servers[i+1:]...)
            lb.healthChecker.RemoveServer(serverID)
            break
        }
    }
}

func (lb *LoadBalancer) GetServer() *Server {
    lb.mutex.RLock()
    defer lb.mutex.RUnlock()
    
    return lb.algorithm.SelectServer(lb.servers)
}

func (lb *LoadBalancer) StartHealthChecking() {
    go lb.healthChecker.Start()
}

func (lb *LoadBalancer) StopHealthChecking() {
    lb.healthChecker.Stop()
}
```

## 🚀 Advanced Network Security

### Network Security Implementation
```go
// Advanced Network Security Implementation
package security

import (
    "crypto/tls"
    "crypto/x509"
    "fmt"
    "net"
    "time"
)

type NetworkSecurity struct {
    tlsConfig    *tls.Config
    firewall     *Firewall
    rateLimiter  *RateLimiter
    ipWhitelist  *IPWhitelist
    ipBlacklist  *IPBlacklist
}

type Firewall struct {
    rules []FirewallRule
    mutex sync.RWMutex
}

type FirewallRule struct {
    ID          string    `json:"id"`
    Action      string    `json:"action"` // "allow", "deny"
    Protocol    string    `json:"protocol"`
    SourceIP    string    `json:"source_ip"`
    DestIP      string    `json:"dest_ip"`
    SourcePort  int       `json:"source_port"`
    DestPort    int       `json:"dest_port"`
    Priority    int       `json:"priority"`
    Enabled     bool      `json:"enabled"`
    CreatedAt   time.Time `json:"created_at"`
}

type RateLimiter struct {
    limits map[string]*RateLimit
    mutex  sync.RWMutex
}

type RateLimit struct {
    Requests   int           `json:"requests"`
    Window     time.Duration `json:"window"`
    Burst      int           `json:"burst"`
    LastReset  time.Time     `json:"last_reset"`
    Count      int           `json:"count"`
}

type IPWhitelist struct {
    ips   map[string]bool
    mutex sync.RWMutex
}

type IPBlacklist struct {
    ips   map[string]bool
    mutex sync.RWMutex
}

func NewNetworkSecurity() *NetworkSecurity {
    return &NetworkSecurity{
        tlsConfig:   createTLSConfig(),
        firewall:    NewFirewall(),
        rateLimiter: NewRateLimiter(),
        ipWhitelist: NewIPWhitelist(),
        ipBlacklist: NewIPBlacklist(),
    }
}

func (ns *NetworkSecurity) SecureConnection(conn net.Conn) (net.Conn, error) {
    // Check IP whitelist/blacklist
    if !ns.isIPAllowed(conn.RemoteAddr()) {
        conn.Close()
        return nil, fmt.Errorf("IP not allowed")
    }
    
    // Check rate limiting
    if !ns.rateLimiter.Allow(conn.RemoteAddr().String()) {
        conn.Close()
        return nil, fmt.Errorf("rate limit exceeded")
    }
    
    // Check firewall rules
    if !ns.firewall.AllowConnection(conn) {
        conn.Close()
        return nil, fmt.Errorf("connection blocked by firewall")
    }
    
    // Wrap connection with TLS
    tlsConn := tls.Server(conn, ns.tlsConfig)
    
    // Perform TLS handshake
    if err := tlsConn.Handshake(); err != nil {
        tlsConn.Close()
        return nil, fmt.Errorf("TLS handshake failed: %w", err)
    }
    
    return tlsConn, nil
}

func (ns *NetworkSecurity) isIPAllowed(addr net.Addr) bool {
    ip, _, err := net.SplitHostPort(addr.String())
    if err != nil {
        return false
    }
    
    // Check blacklist first
    if ns.ipBlacklist.IsBlacklisted(ip) {
        return false
    }
    
    // Check whitelist
    if ns.ipWhitelist.IsWhitelisted(ip) {
        return true
    }
    
    // Default deny
    return false
}

func createTLSConfig() *tls.Config {
    return &tls.Config{
        MinVersion:               tls.VersionTLS12,
        CurvePreferences:         []tls.CurveID{tls.CurveP521, tls.CurveP384, tls.CurveP256},
        PreferServerCipherSuites: true,
        CipherSuites: []uint16{
            tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
            tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
            tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
        },
        ClientAuth: tls.RequireAndVerifyClientCert,
        VerifyPeerCertificate: func(rawCerts [][]byte, verifiedChains [][]*x509.Certificate) error {
            // Custom certificate verification logic
            return nil
        },
    }
}

// Firewall Implementation
func NewFirewall() *Firewall {
    return &Firewall{
        rules: []FirewallRule{},
    }
}

func (fw *Firewall) AddRule(rule *FirewallRule) {
    fw.mutex.Lock()
    defer fw.mutex.Unlock()
    
    rule.CreatedAt = time.Now()
    fw.rules = append(fw.rules, *rule)
    
    // Sort rules by priority
    fw.sortRules()
}

func (fw *Firewall) AllowConnection(conn net.Conn) bool {
    fw.mutex.RLock()
    defer fw.mutex.RUnlock()
    
    addr := conn.RemoteAddr()
    ip, port, err := net.SplitHostPort(addr.String())
    if err != nil {
        return false
    }
    
    for _, rule := range fw.rules {
        if !rule.Enabled {
            continue
        }
        
        if fw.ruleMatches(rule, ip, port) {
            return rule.Action == "allow"
        }
    }
    
    // Default deny
    return false
}

func (fw *Firewall) ruleMatches(rule FirewallRule, ip, port string) bool {
    // Check source IP
    if rule.SourceIP != "" && rule.SourceIP != ip {
        return false
    }
    
    // Check source port
    if rule.SourcePort != 0 {
        if portInt := parsePort(port); portInt != rule.SourcePort {
            return false
        }
    }
    
    return true
}

func (fw *Firewall) sortRules() {
    // Sort by priority (higher priority first)
    for i := 0; i < len(fw.rules)-1; i++ {
        for j := i + 1; j < len(fw.rules); j++ {
            if fw.rules[i].Priority < fw.rules[j].Priority {
                fw.rules[i], fw.rules[j] = fw.rules[j], fw.rules[i]
            }
        }
    }
}

func parsePort(portStr string) int {
    // Simple port parsing - in practice, you'd use strconv.Atoi
    return 0
}

// Rate Limiter Implementation
func NewRateLimiter() *RateLimiter {
    return &RateLimiter{
        limits: make(map[string]*RateLimit),
    }
}

func (rl *RateLimiter) Allow(identifier string) bool {
    rl.mutex.Lock()
    defer rl.mutex.Unlock()
    
    limit, exists := rl.limits[identifier]
    if !exists {
        // Create default limit
        limit = &RateLimit{
            Requests:  100,
            Window:    1 * time.Minute,
            Burst:     10,
            LastReset: time.Now(),
            Count:     0,
        }
        rl.limits[identifier] = limit
    }
    
    // Check if window has expired
    if time.Since(limit.LastReset) > limit.Window {
        limit.Count = 0
        limit.LastReset = time.Now()
    }
    
    // Check if limit exceeded
    if limit.Count >= limit.Requests {
        return false
    }
    
    limit.Count++
    return true
}

func (rl *RateLimiter) SetLimit(identifier string, requests int, window time.Duration, burst int) {
    rl.mutex.Lock()
    defer rl.mutex.Unlock()
    
    rl.limits[identifier] = &RateLimit{
        Requests:  requests,
        Window:    window,
        Burst:     burst,
        LastReset: time.Now(),
        Count:     0,
    }
}

// IP Whitelist Implementation
func NewIPWhitelist() *IPWhitelist {
    return &IPWhitelist{
        ips: make(map[string]bool),
    }
}

func (iwl *IPWhitelist) AddIP(ip string) {
    iwl.mutex.Lock()
    defer iwl.mutex.Unlock()
    
    iwl.ips[ip] = true
}

func (iwl *IPWhitelist) RemoveIP(ip string) {
    iwl.mutex.Lock()
    defer iwl.mutex.Unlock()
    
    delete(iwl.ips, ip)
}

func (iwl *IPWhitelist) IsWhitelisted(ip string) bool {
    iwl.mutex.RLock()
    defer iwl.mutex.RUnlock()
    
    return iwl.ips[ip]
}

// IP Blacklist Implementation
func NewIPBlacklist() *IPBlacklist {
    return &IPBlacklist{
        ips: make(map[string]bool),
    }
}

func (ibl *IPBlacklist) AddIP(ip string) {
    ibl.mutex.Lock()
    defer ibl.mutex.Unlock()
    
    ibl.ips[ip] = true
}

func (ibl *IPBlacklist) RemoveIP(ip string) {
    ibl.mutex.Lock()
    defer ibl.mutex.Unlock()
    
    delete(ibl.ips, ip)
}

func (ibl *IPBlacklist) IsBlacklisted(ip string) bool {
    ibl.mutex.RLock()
    defer ibl.mutex.RUnlock()
    
    return ibl.ips[ip]
}
```

## 🎯 Best Practices

### Network Design Principles
1. **Defense in Depth**: Implement multiple layers of security
2. **Least Privilege**: Grant minimum necessary network access
3. **Segmentation**: Isolate network segments
4. **Monitoring**: Implement comprehensive network monitoring
5. **Redundancy**: Design for high availability

### Performance Optimization
1. **Connection Pooling**: Use connection pools for efficiency
2. **Load Balancing**: Distribute load across multiple servers
3. **Caching**: Implement network-level caching
4. **Compression**: Use compression for data transfer
5. **Protocol Optimization**: Choose appropriate protocols

### Security Best Practices
1. **Encryption**: Encrypt all network communications
2. **Authentication**: Implement strong authentication
3. **Authorization**: Control access to network resources
4. **Monitoring**: Monitor network traffic and events
5. **Incident Response**: Have a clear incident response plan

---

**Last Updated**: December 2024  
**Category**: Advanced Networking Comprehensive  
**Complexity**: Expert Level
