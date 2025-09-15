# 🌐 Networking Fundamentals for Backend Engineers

## Table of Contents
1. [OSI Model](#osi-model)
2. [TCP/IP Protocol Suite](#tcpip-protocol-suite)
3. [HTTP/HTTPS](#httphttps)
4. [DNS](#dns)
5. [Load Balancing](#load-balancing)
6. [CDN](#cdn)
7. [WebSockets](#websockets)
8. [Network Security](#network-security)
9. [Performance Optimization](#performance-optimization)
10. [Interview Questions](#interview-questions)

## OSI Model

### 7-Layer Architecture

```
Application Layer    (7) - HTTP, HTTPS, FTP, SMTP
Presentation Layer  (6) - SSL/TLS, Encryption
Session Layer       (5) - NetBIOS, RPC
Transport Layer     (4) - TCP, UDP
Network Layer       (3) - IP, ICMP, ARP
Data Link Layer     (2) - Ethernet, WiFi
Physical Layer      (1) - Cables, Radio waves
```

### Key Concepts
- **Encapsulation**: Data moves down layers, each adding headers
- **Decapsulation**: Data moves up layers, each removing headers
- **Protocol Data Units (PDUs)**:
  - Application: Data
  - Transport: Segments (TCP) / Datagrams (UDP)
  - Network: Packets
  - Data Link: Frames

## TCP/IP Protocol Suite

### TCP (Transmission Control Protocol)

**Characteristics:**
- Connection-oriented
- Reliable delivery
- Ordered delivery
- Error checking
- Flow control
- Congestion control

**TCP Three-Way Handshake:**
```
Client                    Server
  |                         |
  |-------- SYN ----------->|
  |<------- SYN-ACK --------|
  |-------- ACK ----------->|
  |                         |
```

**TCP Four-Way Handshake (Connection Termination):**
```
Client                    Server
  |                         |
  |-------- FIN ----------->|
  |<------- ACK ------------|
  |<------- FIN ------------|
  |-------- ACK ----------->|
  |                         |
```

### UDP (User Datagram Protocol)

**Characteristics:**
- Connectionless
- Unreliable delivery
- No ordering guarantees
- Lower overhead
- Faster transmission

**Use Cases:**
- Real-time applications (video streaming)
- DNS queries
- Online gaming
- Live broadcasting

### IP (Internet Protocol)

**IPv4:**
- 32-bit addresses
- 4.3 billion addresses
- Dotted decimal notation (192.168.1.1)

**IPv6:**
- 128-bit addresses
- 340 undecillion addresses
- Hexadecimal notation (2001:0db8:85a3::8a2e:0370:7334)

## HTTP/HTTPS

### HTTP (HyperText Transfer Protocol)

**Versions:**
- HTTP/1.0: One request per connection
- HTTP/1.1: Persistent connections, pipelining
- HTTP/2: Multiplexing, server push, header compression
- HTTP/3: QUIC protocol, UDP-based

**Request Methods:**
- GET: Retrieve data
- POST: Submit data
- PUT: Update resource
- DELETE: Remove resource
- PATCH: Partial update
- HEAD: Get headers only
- OPTIONS: Get allowed methods

**Status Codes:**
- 1xx: Informational
- 2xx: Success (200 OK, 201 Created)
- 3xx: Redirection (301 Moved, 304 Not Modified)
- 4xx: Client Error (400 Bad Request, 404 Not Found)
- 5xx: Server Error (500 Internal Server Error, 503 Service Unavailable)

### HTTPS (HTTP Secure)

**Security Features:**
- Encryption using TLS/SSL
- Data integrity verification
- Server authentication
- Client authentication (optional)

**TLS Handshake:**
```
Client                    Server
  |                         |
  |-------- ClientHello --->|
  |<------- ServerHello ----|
  |<------- Certificate ----|
  |<------- ServerDone -----|
  |-------- ClientKeyExch ->|
  |-------- ChangeCipherSpec|
  |-------- Finished ------->|
  |<------- ChangeCipherSpec|
  |<------- Finished -------|
  |                         |
```

## DNS (Domain Name System)

### How DNS Works

```
1. Browser queries local DNS cache
2. If not found, queries recursive DNS server
3. Recursive server queries root nameservers
4. Root nameservers direct to TLD nameservers
5. TLD nameservers direct to authoritative nameservers
6. Authoritative nameservers return IP address
7. Response cached and returned to browser
```

### DNS Record Types

- **A**: IPv4 address
- **AAAA**: IPv6 address
- **CNAME**: Canonical name (alias)
- **MX**: Mail exchange
- **NS**: Name server
- **TXT**: Text record
- **SRV**: Service record

### DNS Caching

**TTL (Time To Live):**
- Controls how long DNS records are cached
- Typical values: 300s (5 min) to 86400s (24 hours)
- Shorter TTL for critical changes
- Longer TTL for stability

## Load Balancing

### Types of Load Balancers

**Layer 4 (Transport Layer):**
- Routes based on IP and port
- Faster processing
- No application awareness
- Examples: HAProxy, F5

**Layer 7 (Application Layer):**
- Routes based on content
- More intelligent routing
- Can handle SSL termination
- Examples: NGINX, AWS ALB

### Load Balancing Algorithms

1. **Round Robin**: Requests distributed evenly
2. **Weighted Round Robin**: Servers with different capacities
3. **Least Connections**: Route to server with fewest active connections
4. **IP Hash**: Route based on client IP
5. **Least Response Time**: Route to fastest responding server

### Load Balancer Configurations

**Active-Passive:**
- One active, one standby
- Failover on active failure
- Simple but underutilized

**Active-Active:**
- Multiple active load balancers
- Better resource utilization
- More complex configuration

## CDN (Content Delivery Network)

### How CDN Works

```
1. User requests content
2. CDN checks edge server cache
3. If cache miss, fetches from origin
4. Content cached at edge server
5. Future requests served from edge
```

### CDN Benefits

- **Reduced Latency**: Content served from nearby servers
- **Reduced Origin Load**: Offloads traffic from origin servers
- **Better Performance**: Optimized delivery
- **Global Reach**: Worldwide content distribution

### CDN Providers

- **CloudFlare**: Security-focused CDN
- **AWS CloudFront**: Amazon's CDN service
- **Google Cloud CDN**: Google's CDN offering
- **Azure CDN**: Microsoft's CDN service

## WebSockets

### WebSocket vs HTTP

**HTTP:**
- Request-response model
- Stateless
- Higher overhead
- One-way communication

**WebSocket:**
- Full-duplex communication
- Persistent connection
- Lower overhead
- Real-time data exchange

### WebSocket Handshake

```
Client Request:
GET /chat HTTP/1.1
Host: server.example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13

Server Response:
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
```

## Network Security

### Common Attacks

**DDoS (Distributed Denial of Service):**
- Multiple sources attack target
- Overwhelms server resources
- Mitigation: Rate limiting, CDN, DDoS protection

**Man-in-the-Middle (MITM):**
- Attacker intercepts communication
- Prevention: HTTPS, certificate pinning

**DNS Spoofing:**
- Malicious DNS responses
- Prevention: DNSSEC, DNS over HTTPS

### Security Measures

**Firewalls:**
- Network-level filtering
- Stateful vs stateless
- Application-level filtering

**VPN (Virtual Private Network):**
- Encrypted tunnel
- Remote access security
- Site-to-site connections

**Intrusion Detection Systems (IDS):**
- Monitor network traffic
- Detect suspicious activity
- Alert on potential threats

## Performance Optimization

### Network Optimization Techniques

**HTTP/2 Features:**
- Multiplexing: Multiple requests over single connection
- Server Push: Proactive content delivery
- Header Compression: Reduced overhead
- Binary Protocol: More efficient parsing

**Caching Strategies:**
- Browser caching
- CDN caching
- Application-level caching
- Database query caching

**Compression:**
- Gzip compression
- Brotli compression
- Image optimization
- Minification

### Monitoring and Metrics

**Key Metrics:**
- Latency (RTT)
- Throughput
- Packet loss
- Jitter
- Connection errors

**Tools:**
- Ping: Basic connectivity test
- Traceroute: Path analysis
- Wireshark: Packet analysis
- Netstat: Network statistics

## Interview Questions

### Basic Concepts

1. **Explain the difference between TCP and UDP.**
2. **What happens when you type a URL in your browser?**
3. **Describe the OSI model and its layers.**
4. **What is the difference between HTTP and HTTPS?**
5. **How does DNS resolution work?**

### Advanced Topics

1. **How would you design a load balancer?**
2. **Explain the TCP three-way handshake.**
3. **What are the benefits of using a CDN?**
4. **How do you handle network failures in a distributed system?**
5. **Describe WebSocket vs HTTP polling.**

### System Design

1. **Design a global CDN system.**
2. **How would you implement rate limiting?**
3. **Design a real-time chat system.**
4. **How would you handle DDoS attacks?**
5. **Design a distributed caching system.**

## Go Implementation Examples

### HTTP Client

```go
package main

import (
    "fmt"
    "io"
    "net/http"
    "time"
)

func main() {
    client := &http.Client{
        Timeout: 10 * time.Second,
    }
    
    resp, err := client.Get("https://api.example.com/data")
    if err != nil {
        panic(err)
    }
    defer resp.Body.Close()
    
    body, err := io.ReadAll(resp.Body)
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("Status: %s\n", resp.Status)
    fmt.Printf("Body: %s\n", string(body))
}
```

### HTTP Server

```go
package main

import (
    "fmt"
    "log"
    "net/http"
)

func handler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Hello, %s!", r.URL.Path[1:])
}

func main() {
    http.HandleFunc("/", handler)
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

### WebSocket Server

```go
package main

import (
    "log"
    "net/http"
    
    "github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
    CheckOrigin: func(r *http.Request) bool {
        return true
    },
}

func handleWebSocket(w http.ResponseWriter, r *http.Request) {
    conn, err := upgrader.Upgrade(w, r, nil)
    if err != nil {
        log.Println("Upgrade failed:", err)
        return
    }
    defer conn.Close()
    
    for {
        messageType, message, err := conn.ReadMessage()
        if err != nil {
            log.Println("Read failed:", err)
            break
        }
        
        err = conn.WriteMessage(messageType, message)
        if err != nil {
            log.Println("Write failed:", err)
            break
        }
    }
}

func main() {
    http.HandleFunc("/ws", handleWebSocket)
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

## Conclusion

Understanding networking fundamentals is crucial for backend engineers. This knowledge helps in:

- Designing scalable systems
- Troubleshooting network issues
- Optimizing application performance
- Making informed architectural decisions
- Preparing for technical interviews

Master these concepts and practice implementing them in your projects to become a well-rounded backend engineer.


continue

continue

continue

continue