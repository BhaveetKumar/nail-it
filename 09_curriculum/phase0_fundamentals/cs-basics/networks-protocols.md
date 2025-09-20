# Networks & Protocols

## Table of Contents

1. [Overview](#overview)
2. [TCP/IP Protocol Stack](#tcpip-protocol-stack)
3. [HTTP and Web Protocols](#http-and-web-protocols)
4. [Network Security](#network-security)
5. [Distributed Systems Basics](#distributed-systems-basics)
6. [Implementations](#implementations)
7. [Follow-up Questions](#follow-up-questions)
8. [Sources](#sources)
9. [Projects](#projects)

## Overview

### Learning Objectives

- Understand TCP/IP protocol stack and networking fundamentals
- Master HTTP and web protocols
- Learn network security concepts and implementation
- Apply distributed systems principles
- Implement network protocols in code

### What are Networks & Protocols?

Networks and protocols are the foundation of modern distributed computing, enabling communication between devices and systems across the internet.

## TCP/IP Protocol Stack

### 1. Protocol Stack Implementation

#### TCP/IP Stack Simulator
```go
package main

import (
    "fmt"
    "net"
    "sync"
    "time"
)

type ProtocolLayer int

const (
    APPLICATION ProtocolLayer = iota
    TRANSPORT
    NETWORK
    DATA_LINK
    PHYSICAL
)

type Packet struct {
    Data        []byte
    SourceIP    string
    DestIP      string
    SourcePort  int
    DestPort    int
    Protocol    string
    Timestamp   time.Time
    TTL         int
    Checksum    uint16
}

type NetworkInterface struct {
    Name        string
    IPAddress   string
    MACAddress  string
    SubnetMask  string
    Gateway     string
    IsUp        bool
    mutex       sync.RWMutex
}

func NewNetworkInterface(name, ip, mac, subnet, gateway string) *NetworkInterface {
    return &NetworkInterface{
        Name:       name,
        IPAddress:  ip,
        MACAddress: mac,
        SubnetMask: subnet,
        Gateway:    gateway,
        IsUp:       true,
    }
}

func (ni *NetworkInterface) SendPacket(packet *Packet) error {
    ni.mutex.RLock()
    defer ni.mutex.RUnlock()
    
    if !ni.IsUp {
        return fmt.Errorf("interface %s is down", ni.Name)
    }
    
    fmt.Printf("Interface %s sending packet from %s:%d to %s:%d\n",
        ni.Name, packet.SourceIP, packet.SourcePort, packet.DestIP, packet.DestPort)
    
    // Simulate packet transmission
    time.Sleep(10 * time.Millisecond)
    
    return nil
}

func (ni *NetworkInterface) ReceivePacket() *Packet {
    ni.mutex.RLock()
    defer ni.mutex.RUnlock()
    
    if !ni.IsUp {
        return nil
    }
    
    // Simulate packet reception
    time.Sleep(5 * time.Millisecond)
    
    return &Packet{
        Data:       []byte("Received data"),
        SourceIP:   "192.168.1.100",
        DestIP:     ni.IPAddress,
        SourcePort: 8080,
        DestPort:   9090,
        Protocol:   "TCP",
        Timestamp:  time.Now(),
        TTL:        64,
        Checksum:   0x1234,
    }
}

type TCPConnection struct {
    SourceIP      string
    SourcePort    int
    DestIP        string
    DestPort      int
    State         string
    SequenceNum   uint32
    AckNum        uint32
    WindowSize    uint16
    mutex         sync.RWMutex
}

func NewTCPConnection(srcIP string, srcPort int, destIP string, destPort int) *TCPConnection {
    return &TCPConnection{
        SourceIP:    srcIP,
        SourcePort:  srcPort,
        DestIP:      destIP,
        DestPort:    destPort,
        State:       "CLOSED",
        SequenceNum: 0,
        AckNum:      0,
        WindowSize:  1024,
    }
}

func (conn *TCPConnection) Connect() error {
    conn.mutex.Lock()
    defer conn.mutex.Unlock()
    
    fmt.Printf("TCP: Sending SYN from %s:%d to %s:%d\n",
        conn.SourceIP, conn.SourcePort, conn.DestIP, conn.DestPort)
    
    conn.State = "SYN_SENT"
    conn.SequenceNum++
    
    // Simulate SYN-ACK
    time.Sleep(50 * time.Millisecond)
    
    fmt.Printf("TCP: Received SYN-ACK from %s:%d\n", conn.DestIP, conn.DestPort)
    conn.State = "ESTABLISHED"
    conn.AckNum++
    
    return nil
}

func (conn *TCPConnection) Send(data []byte) error {
    conn.mutex.Lock()
    defer conn.mutex.Unlock()
    
    if conn.State != "ESTABLISHED" {
        return fmt.Errorf("connection not established")
    }
    
    fmt.Printf("TCP: Sending %d bytes from %s:%d to %s:%d (seq: %d)\n",
        len(data), conn.SourceIP, conn.SourcePort, conn.DestIP, conn.DestPort, conn.SequenceNum)
    
    conn.SequenceNum += uint32(len(data))
    
    // Simulate ACK
    time.Sleep(20 * time.Millisecond)
    conn.AckNum++
    
    return nil
}

func (conn *TCPConnection) Close() error {
    conn.mutex.Lock()
    defer conn.mutex.Unlock()
    
    fmt.Printf("TCP: Sending FIN from %s:%d to %s:%d\n",
        conn.SourceIP, conn.SourcePort, conn.DestIP, conn.DestPort)
    
    conn.State = "FIN_WAIT_1"
    
    // Simulate FIN-ACK
    time.Sleep(30 * time.Millisecond)
    
    conn.State = "CLOSED"
    return nil
}

type NetworkStack struct {
    Interfaces map[string]*NetworkInterface
    Connections map[string]*TCPConnection
    mutex      sync.RWMutex
}

func NewNetworkStack() *NetworkStack {
    return &NetworkStack{
        Interfaces:  make(map[string]*NetworkInterface),
        Connections: make(map[string]*TCPConnection),
    }
}

func (ns *NetworkStack) AddInterface(iface *NetworkInterface) {
    ns.mutex.Lock()
    defer ns.mutex.Unlock()
    
    ns.Interfaces[iface.Name] = iface
    fmt.Printf("Added interface: %s (%s)\n", iface.Name, iface.IPAddress)
}

func (ns *NetworkStack) CreateConnection(srcIP string, srcPort int, destIP string, destPort int) *TCPConnection {
    ns.mutex.Lock()
    defer ns.mutex.Unlock()
    
    conn := NewTCPConnection(srcIP, srcPort, destIP, destPort)
    connKey := fmt.Sprintf("%s:%d->%s:%d", srcIP, srcPort, destIP, destPort)
    ns.Connections[connKey] = conn
    
    return conn
}

func (ns *NetworkStack) SendData(srcIP string, srcPort int, destIP string, destPort int, data []byte) error {
    connKey := fmt.Sprintf("%s:%d->%s:%d", srcIP, srcPort, destIP, destPort)
    
    ns.mutex.RLock()
    conn, exists := ns.Connections[connKey]
    ns.mutex.RUnlock()
    
    if !exists {
        conn = ns.CreateConnection(srcIP, srcPort, destIP, destPort)
        if err := conn.Connect(); err != nil {
            return err
        }
    }
    
    return conn.Send(data)
}

func (ns *NetworkStack) PrintStatus() {
    ns.mutex.RLock()
    defer ns.mutex.RUnlock()
    
    fmt.Println("\nNetwork Stack Status:")
    fmt.Println("====================")
    
    fmt.Println("Interfaces:")
    for name, iface := range ns.Interfaces {
        fmt.Printf("  %s: %s (%s) - %s\n", name, iface.IPAddress, iface.MACAddress, 
            map[bool]string{true: "UP", false: "DOWN"}[iface.IsUp])
    }
    
    fmt.Println("\nConnections:")
    for key, conn := range ns.Connections {
        fmt.Printf("  %s: %s (seq: %d, ack: %d)\n", key, conn.State, conn.SequenceNum, conn.AckNum)
    }
}

func main() {
    stack := NewNetworkStack()
    
    // Add network interfaces
    eth0 := NewNetworkInterface("eth0", "192.168.1.10", "00:11:22:33:44:55", "255.255.255.0", "192.168.1.1")
    wlan0 := NewNetworkInterface("wlan0", "10.0.0.5", "aa:bb:cc:dd:ee:ff", "255.255.255.0", "10.0.0.1")
    
    stack.AddInterface(eth0)
    stack.AddInterface(wlan0)
    
    // Create TCP connection
    conn := stack.CreateConnection("192.168.1.10", 8080, "192.168.1.100", 9090)
    conn.Connect()
    
    // Send data
    data := []byte("Hello, World!")
    stack.SendData("192.168.1.10", 8080, "192.168.1.100", 9090, data)
    
    // Close connection
    conn.Close()
    
    stack.PrintStatus()
}
```

### 2. IP Routing Implementation

#### Simple Router
```go
package main

import (
    "fmt"
    "net"
    "sort"
    "sync"
)

type Route struct {
    Destination string
    Gateway     string
    Interface   string
    Metric      int
    Mask        string
}

type RoutingTable struct {
    Routes []Route
    mutex  sync.RWMutex
}

func NewRoutingTable() *RoutingTable {
    return &RoutingTable{
        Routes: make([]Route, 0),
    }
}

func (rt *RoutingTable) AddRoute(dest, gateway, iface, mask string, metric int) {
    rt.mutex.Lock()
    defer rt.mutex.Unlock()
    
    route := Route{
        Destination: dest,
        Gateway:     gateway,
        Interface:   iface,
        Metric:      metric,
        Mask:        mask,
    }
    
    rt.Routes = append(rt.Routes, route)
    
    // Sort by metric (lower is better)
    sort.Slice(rt.Routes, func(i, j int) bool {
        return rt.Routes[i].Metric < rt.Routes[j].Metric
    })
    
    fmt.Printf("Added route: %s/%s via %s on %s (metric: %d)\n", dest, mask, gateway, iface, metric)
}

func (rt *RoutingTable) FindRoute(destIP string) *Route {
    rt.mutex.RLock()
    defer rt.mutex.RUnlock()
    
    dest := net.ParseIP(destIP)
    if dest == nil {
        return nil
    }
    
    for _, route := range rt.Routes {
        if rt.isInSubnet(dest, route.Destination, route.Mask) {
            return &route
        }
    }
    
    return nil
}

func (rt *RoutingTable) isInSubnet(ip net.IP, network, mask string) bool {
    _, ipNet, err := net.ParseCIDR(network + "/" + mask)
    if err != nil {
        return false
    }
    return ipNet.Contains(ip)
}

func (rt *RoutingTable) PrintTable() {
    rt.mutex.RLock()
    defer rt.mutex.RUnlock()
    
    fmt.Println("\nRouting Table:")
    fmt.Println("==============")
    fmt.Printf("%-15s %-15s %-10s %-10s %s\n", "Destination", "Gateway", "Interface", "Metric", "Mask")
    fmt.Println("------------------------------------------------------------------------")
    
    for _, route := range rt.Routes {
        fmt.Printf("%-15s %-15s %-10s %-10d %s\n",
            route.Destination, route.Gateway, route.Interface, route.Metric, route.Mask)
    }
}

type Router struct {
    Name          string
    Interfaces    map[string]*NetworkInterface
    RoutingTable  *RoutingTable
    mutex         sync.RWMutex
}

func NewRouter(name string) *Router {
    return &Router{
        Name:         name,
        Interfaces:   make(map[string]*NetworkInterface),
        RoutingTable: NewRoutingTable(),
    }
}

func (r *Router) AddInterface(iface *NetworkInterface) {
    r.mutex.Lock()
    defer r.mutex.Unlock()
    
    r.Interfaces[iface.Name] = iface
    
    // Add connected route
    r.RoutingTable.AddRoute(iface.IPAddress, "0.0.0.0", iface.Name, "32", 0)
}

func (r *Router) RoutePacket(packet *Packet) error {
    r.mutex.RLock()
    defer r.mutex.RUnlock()
    
    route := r.RoutingTable.FindRoute(packet.DestIP)
    if route == nil {
        return fmt.Errorf("no route to %s", packet.DestIP)
    }
    
    iface, exists := r.Interfaces[route.Interface]
    if !exists {
        return fmt.Errorf("interface %s not found", route.Interface)
    }
    
    fmt.Printf("Router %s: Routing packet to %s via %s on %s\n",
        r.Name, packet.DestIP, route.Gateway, route.Interface)
    
    return iface.SendPacket(packet)
}

func (r *Router) PrintStatus() {
    r.mutex.RLock()
    defer r.mutex.RUnlock()
    
    fmt.Printf("\nRouter %s Status:\n", r.Name)
    fmt.Println("==================")
    
    fmt.Println("Interfaces:")
    for name, iface := range r.Interfaces {
        fmt.Printf("  %s: %s (%s)\n", name, iface.IPAddress, iface.MACAddress)
    }
    
    r.RoutingTable.PrintTable()
}

func main() {
    router := NewRouter("Router1")
    
    // Add interfaces
    eth0 := NewNetworkInterface("eth0", "192.168.1.1", "00:11:22:33:44:55", "255.255.255.0", "0.0.0.0")
    eth1 := NewNetworkInterface("eth1", "10.0.0.1", "aa:bb:cc:dd:ee:ff", "255.255.255.0", "0.0.0.0")
    
    router.AddInterface(eth0)
    router.AddInterface(eth1)
    
    // Add routes
    router.RoutingTable.AddRoute("192.168.1.0", "0.0.0.0", "eth0", "24", 0)
    router.RoutingTable.AddRoute("10.0.0.0", "0.0.0.0", "eth1", "24", 0)
    router.RoutingTable.AddRoute("0.0.0.0", "192.168.1.254", "eth0", "0", 1)
    
    // Test routing
    packet := &Packet{
        Data:       []byte("Test packet"),
        SourceIP:   "192.168.1.10",
        DestIP:     "10.0.0.5",
        SourcePort: 8080,
        DestPort:   9090,
        Protocol:   "TCP",
        TTL:        64,
    }
    
    router.RoutePacket(packet)
    router.PrintStatus()
}
```

## HTTP and Web Protocols

### 1. HTTP Client Implementation

#### HTTP Client
```go
package main

import (
    "fmt"
    "io"
    "net/http"
    "net/url"
    "strings"
    "time"
)

type HTTPClient struct {
    Client      *http.Client
    BaseURL     string
    Headers     map[string]string
    Timeout     time.Duration
}

func NewHTTPClient(baseURL string, timeout time.Duration) *HTTPClient {
    return &HTTPClient{
        Client: &http.Client{
            Timeout: timeout,
        },
        BaseURL: baseURL,
        Headers: make(map[string]string),
        Timeout: timeout,
    }
}

func (c *HTTPClient) SetHeader(key, value string) {
    c.Headers[key] = value
}

func (c *HTTPClient) Get(path string) (*HTTPResponse, error) {
    return c.Request("GET", path, nil, nil)
}

func (c *HTTPClient) Post(path string, data map[string]string) (*HTTPResponse, error) {
    values := url.Values{}
    for k, v := range data {
        values.Set(k, v)
    }
    
    return c.Request("POST", path, strings.NewReader(values.Encode()), map[string]string{
        "Content-Type": "application/x-www-form-urlencoded",
    })
}

func (c *HTTPClient) PostJSON(path string, jsonData string) (*HTTPResponse, error) {
    return c.Request("POST", path, strings.NewReader(jsonData), map[string]string{
        "Content-Type": "application/json",
    })
}

func (c *HTTPClient) Request(method, path string, body io.Reader, headers map[string]string) (*HTTPResponse, error) {
    fullURL := c.BaseURL + path
    req, err := http.NewRequest(method, fullURL, body)
    if err != nil {
        return nil, err
    }
    
    // Set default headers
    for k, v := range c.Headers {
        req.Header.Set(k, v)
    }
    
    // Set request-specific headers
    for k, v := range headers {
        req.Header.Set(k, v)
    }
    
    start := time.Now()
    resp, err := c.Client.Do(req)
    duration := time.Since(start)
    
    if err != nil {
        return nil, err
    }
    
    defer resp.Body.Close()
    
    bodyBytes, err := io.ReadAll(resp.Body)
    if err != nil {
        return nil, err
    }
    
    return &HTTPResponse{
        StatusCode: resp.StatusCode,
        Headers:    resp.Header,
        Body:       string(bodyBytes),
        Duration:   duration,
    }, nil
}

type HTTPResponse struct {
    StatusCode int
    Headers    http.Header
    Body       string
    Duration   time.Duration
}

func (r *HTTPResponse) String() string {
    return fmt.Sprintf("HTTP %d\nDuration: %v\nBody: %s", r.StatusCode, r.Duration, r.Body)
}

func main() {
    // Create HTTP client
    client := NewHTTPClient("https://httpbin.org", 30*time.Second)
    
    // Set default headers
    client.SetHeader("User-Agent", "Go-HTTP-Client/1.0")
    client.SetHeader("Accept", "application/json")
    
    // Test GET request
    fmt.Println("Testing GET request:")
    resp, err := client.Get("/get")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Println(resp.String())
    }
    
    // Test POST request
    fmt.Println("\nTesting POST request:")
    data := map[string]string{
        "name":  "John Doe",
        "email": "john@example.com",
    }
    resp, err = client.Post("/post", data)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Println(resp.String())
    }
    
    // Test JSON POST
    fmt.Println("\nTesting JSON POST request:")
    jsonData := `{"name": "Jane Doe", "email": "jane@example.com"}`
    resp, err = client.PostJSON("/post", jsonData)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Println(resp.String())
    }
}
```

### 2. WebSocket Implementation

#### WebSocket Client
```go
package main

import (
    "fmt"
    "net/http"
    "net/url"
    "time"
)

type WebSocketClient struct {
    URL        string
    Headers    map[string]string
    Connected  bool
    Messages   chan string
    Errors     chan error
}

func NewWebSocketClient(wsURL string) *WebSocketClient {
    return &WebSocketClient{
        URL:       wsURL,
        Headers:   make(map[string]string),
        Connected: false,
        Messages:  make(chan string, 100),
        Errors:    make(chan error, 100),
    }
}

func (ws *WebSocketClient) Connect() error {
    u, err := url.Parse(ws.URL)
    if err != nil {
        return err
    }
    
    // Simulate WebSocket handshake
    fmt.Printf("WebSocket: Connecting to %s\n", u.Host)
    
    // Simulate connection establishment
    time.Sleep(100 * time.Millisecond)
    
    ws.Connected = true
    fmt.Println("WebSocket: Connected successfully")
    
    // Start message handling goroutine
    go ws.handleMessages()
    
    return nil
}

func (ws *WebSocketClient) handleMessages() {
    for ws.Connected {
        // Simulate receiving messages
        time.Sleep(1 * time.Second)
        
        if ws.Connected {
            message := fmt.Sprintf("Message received at %s", time.Now().Format(time.RFC3339))
            select {
            case ws.Messages <- message:
            default:
                // Channel full, skip message
            }
        }
    }
}

func (ws *WebSocketClient) Send(message string) error {
    if !ws.Connected {
        return fmt.Errorf("WebSocket not connected")
    }
    
    fmt.Printf("WebSocket: Sending message: %s\n", message)
    
    // Simulate message sending
    time.Sleep(50 * time.Millisecond)
    
    return nil
}

func (ws *WebSocketClient) Close() error {
    if !ws.Connected {
        return fmt.Errorf("WebSocket not connected")
    }
    
    fmt.Println("WebSocket: Closing connection")
    ws.Connected = false
    
    return nil
}

func (ws *WebSocketClient) Listen() {
    for ws.Connected {
        select {
        case message := <-ws.Messages:
            fmt.Printf("WebSocket: Received: %s\n", message)
        case err := <-ws.Errors:
            fmt.Printf("WebSocket: Error: %v\n", err)
        case <-time.After(5 * time.Second):
            fmt.Println("WebSocket: No messages received in 5 seconds")
        }
    }
}

func main() {
    ws := NewWebSocketClient("ws://localhost:8080/ws")
    
    // Connect
    if err := ws.Connect(); err != nil {
        fmt.Printf("Connection error: %v\n", err)
        return
    }
    
    // Send some messages
    go func() {
        for i := 0; i < 5; i++ {
            message := fmt.Sprintf("Hello from client %d", i+1)
            ws.Send(message)
            time.Sleep(2 * time.Second)
        }
        ws.Close()
    }()
    
    // Listen for messages
    ws.Listen()
}
```

## Network Security

### 1. Encryption Implementation

#### AES Encryption
```go
package main

import (
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "crypto/sha256"
    "fmt"
    "io"
)

type AESEncryption struct {
    Key []byte
}

func NewAESEncryption(key string) *AESEncryption {
    hash := sha256.Sum256([]byte(key))
    return &AESEncryption{
        Key: hash[:],
    }
}

func (ae *AESEncryption) Encrypt(plaintext []byte) ([]byte, error) {
    block, err := aes.NewCipher(ae.Key)
    if err != nil {
        return nil, err
    }
    
    // Create GCM mode
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }
    
    // Create nonce
    nonce := make([]byte, gcm.NonceSize())
    if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
        return nil, err
    }
    
    // Encrypt
    ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)
    
    return ciphertext, nil
}

func (ae *AESEncryption) Decrypt(ciphertext []byte) ([]byte, error) {
    block, err := aes.NewCipher(ae.Key)
    if err != nil {
        return nil, err
    }
    
    // Create GCM mode
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }
    
    // Extract nonce
    nonceSize := gcm.NonceSize()
    if len(ciphertext) < nonceSize {
        return nil, fmt.Errorf("ciphertext too short")
    }
    
    nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]
    
    // Decrypt
    plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
    if err != nil {
        return nil, err
    }
    
    return plaintext, nil
}

func main() {
    // Create encryption instance
    encryption := NewAESEncryption("my-secret-key")
    
    // Encrypt data
    plaintext := "This is a secret message that needs to be encrypted."
    fmt.Printf("Original: %s\n", plaintext)
    
    ciphertext, err := encryption.Encrypt([]byte(plaintext))
    if err != nil {
        fmt.Printf("Encryption error: %v\n", err)
        return
    }
    
    fmt.Printf("Encrypted: %x\n", ciphertext)
    
    // Decrypt data
    decrypted, err := encryption.Decrypt(ciphertext)
    if err != nil {
        fmt.Printf("Decryption error: %v\n", err)
        return
    }
    
    fmt.Printf("Decrypted: %s\n", string(decrypted))
}
```

### 2. TLS/SSL Implementation

#### TLS Client
```go
package main

import (
    "crypto/tls"
    "crypto/x509"
    "fmt"
    "io"
    "net"
    "time"
)

type TLSClient struct {
    Config *tls.Config
    Conn   net.Conn
}

func NewTLSClient() *TLSClient {
    return &TLSClient{
        Config: &tls.Config{
            InsecureSkipVerify: false, // Set to true for testing only
        },
    }
}

func (tc *TLSClient) Connect(host, port string) error {
    address := host + ":" + port
    
    // Create TCP connection
    conn, err := net.DialTimeout("tcp", address, 30*time.Second)
    if err != nil {
        return err
    }
    
    // Create TLS connection
    tlsConn := tls.Client(conn, tc.Config)
    
    // Perform TLS handshake
    if err := tlsConn.Handshake(); err != nil {
        conn.Close()
        return err
    }
    
    tc.Conn = tlsConn
    
    // Print connection info
    state := tlsConn.ConnectionState()
    fmt.Printf("TLS: Connected to %s\n", address)
    fmt.Printf("TLS: Version: %x\n", state.Version)
    fmt.Printf("TLS: Cipher Suite: %x\n", state.CipherSuite)
    fmt.Printf("TLS: Server Name: %s\n", state.ServerName)
    
    return nil
}

func (tc *TLSClient) Send(data []byte) error {
    if tc.Conn == nil {
        return fmt.Errorf("not connected")
    }
    
    _, err := tc.Conn.Write(data)
    return err
}

func (tc *TLSClient) Receive() ([]byte, error) {
    if tc.Conn == nil {
        return nil, fmt.Errorf("not connected")
    }
    
    buffer := make([]byte, 1024)
    n, err := tc.Conn.Read(buffer)
    if err != nil {
        return nil, err
    }
    
    return buffer[:n], nil
}

func (tc *TLSClient) Close() error {
    if tc.Conn != nil {
        return tc.Conn.Close()
    }
    return nil
}

func (tc *TLSClient) GetCertificateInfo() {
    if tc.Conn == nil {
        fmt.Println("TLS: Not connected")
        return
    }
    
    state := tc.Conn.(*tls.Conn).ConnectionState()
    
    fmt.Println("\nTLS Certificate Information:")
    fmt.Println("===========================")
    
    for i, cert := range state.PeerCertificates {
        fmt.Printf("Certificate %d:\n", i+1)
        fmt.Printf("  Subject: %s\n", cert.Subject)
        fmt.Printf("  Issuer: %s\n", cert.Issuer)
        fmt.Printf("  Not Before: %s\n", cert.NotBefore)
        fmt.Printf("  Not After: %s\n", cert.NotAfter)
        fmt.Printf("  Serial Number: %s\n", cert.SerialNumber)
        fmt.Printf("  Public Key Algorithm: %s\n", cert.PublicKeyAlgorithm)
        fmt.Printf("  Signature Algorithm: %s\n", cert.SignatureAlgorithm)
        fmt.Println()
    }
}

func main() {
    client := NewTLSClient()
    
    // Connect to a TLS server (example: Google)
    if err := client.Connect("www.google.com", "443"); err != nil {
        fmt.Printf("Connection error: %v\n", err)
        return
    }
    
    defer client.Close()
    
    // Get certificate information
    client.GetCertificateInfo()
    
    // Send HTTP request
    request := "GET / HTTP/1.1\r\nHost: www.google.com\r\nConnection: close\r\n\r\n"
    if err := client.Send([]byte(request)); err != nil {
        fmt.Printf("Send error: %v\n", err)
        return
    }
    
    // Receive response
    response, err := client.Receive()
    if err != nil {
        fmt.Printf("Receive error: %v\n", err)
        return
    }
    
    fmt.Printf("Response: %s\n", string(response))
}
```

## Distributed Systems Basics

### 1. Load Balancer Implementation

#### Round Robin Load Balancer
```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Server struct {
    ID       string
    Address  string
    Port     int
    Healthy  bool
    Load     int
    mutex    sync.RWMutex
}

func NewServer(id, address string, port int) *Server {
    return &Server{
        ID:      id,
        Address: address,
        Port:    port,
        Healthy: true,
        Load:    0,
    }
}

func (s *Server) SetHealth(healthy bool) {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    s.Healthy = healthy
}

func (s *Server) IsHealthy() bool {
    s.mutex.RLock()
    defer s.mutex.RUnlock()
    return s.Healthy
}

func (s *Server) IncrementLoad() {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    s.Load++
}

func (s *Server) DecrementLoad() {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    if s.Load > 0 {
        s.Load--
    }
}

func (s *Server) GetLoad() int {
    s.mutex.RLock()
    defer s.mutex.RUnlock()
    return s.Load
}

func (s *Server) String() string {
    s.mutex.RLock()
    defer s.mutex.RUnlock()
    
    status := "DOWN"
    if s.Healthy {
        status = "UP"
    }
    
    return fmt.Sprintf("Server %s (%s:%d) - %s (Load: %d)", 
        s.ID, s.Address, s.Port, status, s.Load)
}

type LoadBalancer struct {
    Servers    []*Server
    Current    int
    mutex      sync.RWMutex
}

func NewLoadBalancer() *LoadBalancer {
    return &LoadBalancer{
        Servers: make([]*Server, 0),
        Current: 0,
    }
}

func (lb *LoadBalancer) AddServer(server *Server) {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()
    
    lb.Servers = append(lb.Servers, server)
    fmt.Printf("Added server: %s\n", server.String())
}

func (lb *LoadBalancer) RemoveServer(serverID string) {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()
    
    for i, server := range lb.Servers {
        if server.ID == serverID {
            lb.Servers = append(lb.Servers[:i], lb.Servers[i+1:]...)
            fmt.Printf("Removed server: %s\n", serverID)
            break
        }
    }
}

func (lb *LoadBalancer) GetNextServer() *Server {
    lb.mutex.Lock()
    defer lb.mutex.Unlock()
    
    if len(lb.Servers) == 0 {
        return nil
    }
    
    // Find next healthy server
    attempts := 0
    for attempts < len(lb.Servers) {
        server := lb.Servers[lb.Current]
        if server.IsHealthy() {
            lb.Current = (lb.Current + 1) % len(lb.Servers)
            return server
        }
        lb.Current = (lb.Current + 1) % len(lb.Servers)
        attempts++
    }
    
    return nil
}

func (lb *LoadBalancer) ForwardRequest(request string) error {
    server := lb.GetNextServer()
    if server == nil {
        return fmt.Errorf("no healthy servers available")
    }
    
    server.IncrementLoad()
    fmt.Printf("Forwarding request '%s' to %s\n", request, server.String())
    
    // Simulate request processing
    time.Sleep(100 * time.Millisecond)
    
    server.DecrementLoad()
    fmt.Printf("Request completed on %s\n", server.String())
    
    return nil
}

func (lb *LoadBalancer) HealthCheck() {
    lb.mutex.RLock()
    defer lb.mutex.RUnlock()
    
    fmt.Println("\nHealth Check Results:")
    fmt.Println("====================")
    
    for _, server := range lb.Servers {
        // Simulate health check
        healthy := server.GetLoad() < 10 // Simple health check based on load
        
        if server.IsHealthy() != healthy {
            server.SetHealth(healthy)
            status := "DOWN"
            if healthy {
                status = "UP"
            }
            fmt.Printf("Server %s status changed to %s\n", server.ID, status)
        }
        
        fmt.Printf("  %s\n", server.String())
    }
}

func (lb *LoadBalancer) PrintStatus() {
    lb.mutex.RLock()
    defer lb.mutex.RUnlock()
    
    fmt.Println("\nLoad Balancer Status:")
    fmt.Println("====================")
    fmt.Printf("Total Servers: %d\n", len(lb.Servers))
    
    healthyCount := 0
    for _, server := range lb.Servers {
        if server.IsHealthy() {
            healthyCount++
        }
    }
    
    fmt.Printf("Healthy Servers: %d\n", healthyCount)
    fmt.Printf("Current Index: %d\n", lb.Current)
    
    fmt.Println("\nServer Details:")
    for _, server := range lb.Servers {
        fmt.Printf("  %s\n", server.String())
    }
}

func main() {
    lb := NewLoadBalancer()
    
    // Add servers
    server1 := NewServer("web1", "192.168.1.10", 8080)
    server2 := NewServer("web2", "192.168.1.11", 8080)
    server3 := NewServer("web3", "192.168.1.12", 8080)
    
    lb.AddServer(server1)
    lb.AddServer(server2)
    lb.AddServer(server3)
    
    // Simulate requests
    for i := 0; i < 10; i++ {
        request := fmt.Sprintf("Request %d", i+1)
        if err := lb.ForwardRequest(request); err != nil {
            fmt.Printf("Error: %v\n", err)
        }
        time.Sleep(50 * time.Millisecond)
    }
    
    // Simulate server failure
    fmt.Println("\nSimulating server failure...")
    server2.SetHealth(false)
    
    // Continue with requests
    for i := 10; i < 15; i++ {
        request := fmt.Sprintf("Request %d", i+1)
        if err := lb.ForwardRequest(request); err != nil {
            fmt.Printf("Error: %v\n", err)
        }
        time.Sleep(50 * time.Millisecond)
    }
    
    // Health check
    lb.HealthCheck()
    
    // Print final status
    lb.PrintStatus()
}
```

## Follow-up Questions

### 1. TCP/IP Protocol Stack
**Q: What's the difference between TCP and UDP?**
A: TCP provides reliable, ordered delivery with error checking and flow control, while UDP provides fast, connectionless delivery without guarantees.

### 2. HTTP and Web Protocols
**Q: How does HTTPS differ from HTTP?**
A: HTTPS adds TLS/SSL encryption to HTTP, providing secure communication over the internet.

### 3. Network Security
**Q: What are the main threats to network security?**
A: Main threats include eavesdropping, man-in-the-middle attacks, denial of service, and data tampering.

## Sources

### Books
- **Computer Networks** by Tanenbaum
- **TCP/IP Illustrated** by Stevens
- **Network Security Essentials** by Stallings

### Online Resources
- **RFC Documents**: Official protocol specifications
- **Coursera**: Computer Networks courses
- **YouTube**: Network engineering tutorials

## Projects

### 1. Network Protocol Analyzer
**Objective**: Build a network packet analyzer
**Requirements**: Packet capture, protocol parsing, traffic analysis
**Deliverables**: Working analyzer with protocol support

### 2. Load Balancer
**Objective**: Implement a load balancing system
**Requirements**: Multiple algorithms, health checking, failover
**Deliverables**: Load balancer with monitoring

### 3. Secure Chat Application
**Objective**: Create an encrypted chat system
**Requirements**: End-to-end encryption, key exchange, message authentication
**Deliverables**: Secure chat application

---

**Next**: [Database Fundamentals](database-fundamentals.md) | **Previous**: [Operating Systems](operating-systems-concepts.md) | **Up**: [Phase 0](README.md)
