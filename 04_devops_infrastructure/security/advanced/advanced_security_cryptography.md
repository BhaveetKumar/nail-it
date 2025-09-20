# 🔐 Advanced Security & Cryptography Guide

## Table of Contents
1. [Cryptographic Fundamentals](#cryptographic-fundamentals)
2. [Symmetric Encryption](#symmetric-encryption)
3. [Asymmetric Encryption](#asymmetric-encryption)
4. [Hash Functions](#hash-functions)
5. [Digital Signatures](#digital-signatures)
6. [Key Management](#key-management)
7. [TLS/SSL Implementation](#tlsssl-implementation)
8. [OAuth 2.0 & JWT](#oauth-20--jwt)
9. [Security Best Practices](#security-best-practices)
10. [Go Implementation Examples](#go-implementation-examples)
11. [Interview Questions](#interview-questions)

## Cryptographic Fundamentals

### Encryption vs Hashing vs Encoding

```go
package main

import (
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "crypto/sha256"
    "encoding/base64"
    "encoding/hex"
    "fmt"
    "io"
)

// Encryption: Reversible transformation of data
func EncryptAES(plaintext []byte, key []byte) ([]byte, error) {
    block, err := aes.NewCipher(key)
    if err != nil {
        return nil, err
    }
    
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }
    
    nonce := make([]byte, gcm.NonceSize())
    if _, err = io.ReadFull(rand.Reader, nonce); err != nil {
        return nil, err
    }
    
    ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)
    return ciphertext, nil
}

func DecryptAES(ciphertext []byte, key []byte) ([]byte, error) {
    block, err := aes.NewCipher(key)
    if err != nil {
        return nil, err
    }
    
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }
    
    nonceSize := gcm.NonceSize()
    if len(ciphertext) < nonceSize {
        return nil, fmt.Errorf("ciphertext too short")
    }
    
    nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]
    plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
    if err != nil {
        return nil, err
    }
    
    return plaintext, nil
}

// Hashing: One-way transformation of data
func HashSHA256(data []byte) string {
    hash := sha256.Sum256(data)
    return hex.EncodeToString(hash[:])
}

func HashWithSalt(data []byte, salt []byte) string {
    combined := append(data, salt...)
    return HashSHA256(combined)
}

// Encoding: Reversible transformation for data representation
func EncodeBase64(data []byte) string {
    return base64.StdEncoding.EncodeToString(data)
}

func DecodeBase64(encoded string) ([]byte, error) {
    return base64.StdEncoding.DecodeString(encoded)
}

// Example usage
func main() {
    // Encryption example
    key := []byte("32-byte-long-key-for-AES-256!")
    plaintext := []byte("Hello, World!")
    
    ciphertext, err := EncryptAES(plaintext, key)
    if err != nil {
        fmt.Printf("Encryption error: %v\n", err)
        return
    }
    
    decrypted, err := DecryptAES(ciphertext, key)
    if err != nil {
        fmt.Printf("Decryption error: %v\n", err)
        return
    }
    
    fmt.Printf("Original: %s\n", string(plaintext))
    fmt.Printf("Encrypted: %x\n", ciphertext)
    fmt.Printf("Decrypted: %s\n", string(decrypted))
    
    // Hashing example
    hash := HashSHA256(plaintext)
    fmt.Printf("SHA256 Hash: %s\n", hash)
    
    // Encoding example
    encoded := EncodeBase64(plaintext)
    decoded, _ := DecodeBase64(encoded)
    fmt.Printf("Base64 Encoded: %s\n", encoded)
    fmt.Printf("Base64 Decoded: %s\n", string(decoded))
}
```

## Symmetric Encryption

### AES Implementation

```go
package main

import (
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "fmt"
    "io"
)

type AESEncryption struct {
    key []byte
}

func NewAESEncryption(key []byte) (*AESEncryption, error) {
    if len(key) != 16 && len(key) != 24 && len(key) != 32 {
        return nil, fmt.Errorf("key must be 16, 24, or 32 bytes")
    }
    
    return &AESEncryption{key: key}, nil
}

func (aes *AESEncryption) Encrypt(plaintext []byte) ([]byte, error) {
    block, err := aes.NewCipher(aes.key)
    if err != nil {
        return nil, err
    }
    
    // Use GCM mode for authenticated encryption
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }
    
    // Generate random nonce
    nonce := make([]byte, gcm.NonceSize())
    if _, err = io.ReadFull(rand.Reader, nonce); err != nil {
        return nil, err
    }
    
    // Encrypt and authenticate
    ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)
    return ciphertext, nil
}

func (aes *AESEncryption) Decrypt(ciphertext []byte) ([]byte, error) {
    block, err := aes.NewCipher(aes.key)
    if err != nil {
        return nil, err
    }
    
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }
    
    nonceSize := gcm.NonceSize()
    if len(ciphertext) < nonceSize {
        return nil, fmt.Errorf("ciphertext too short")
    }
    
    nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]
    plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
    if err != nil {
        return nil, err
    }
    
    return plaintext, nil
}

// CBC Mode Implementation
func (aes *AESEncryption) EncryptCBC(plaintext []byte) ([]byte, error) {
    block, err := aes.NewCipher(aes.key)
    if err != nil {
        return nil, err
    }
    
    // PKCS7 padding
    plaintext = pkcs7Padding(plaintext, aes.BlockSize())
    
    // Generate random IV
    iv := make([]byte, aes.BlockSize())
    if _, err = io.ReadFull(rand.Reader, iv); err != nil {
        return nil, err
    }
    
    // Encrypt
    mode := cipher.NewCBCEncrypter(block, iv)
    ciphertext := make([]byte, len(plaintext))
    mode.CryptBlocks(ciphertext, plaintext)
    
    // Prepend IV to ciphertext
    return append(iv, ciphertext...), nil
}

func (aes *AESEncryption) DecryptCBC(ciphertext []byte) ([]byte, error) {
    block, err := aes.NewCipher(aes.key)
    if err != nil {
        return nil, err
    }
    
    if len(ciphertext) < aes.BlockSize() {
        return nil, fmt.Errorf("ciphertext too short")
    }
    
    iv := ciphertext[:aes.BlockSize()]
    ciphertext = ciphertext[aes.BlockSize():]
    
    if len(ciphertext)%aes.BlockSize() != 0 {
        return nil, fmt.Errorf("ciphertext is not a multiple of the block size")
    }
    
    mode := cipher.NewCBCDecrypter(block, iv)
    plaintext := make([]byte, len(ciphertext))
    mode.CryptBlocks(plaintext, ciphertext)
    
    // Remove PKCS7 padding
    return pkcs7UnPadding(plaintext), nil
}

func (aes *AESEncryption) BlockSize() int {
    return aes.BlockSize()
}

// PKCS7 Padding
func pkcs7Padding(data []byte, blockSize int) []byte {
    padding := blockSize - len(data)%blockSize
    padtext := make([]byte, padding)
    for i := range padtext {
        padtext[i] = byte(padding)
    }
    return append(data, padtext...)
}

func pkcs7UnPadding(data []byte) []byte {
    length := len(data)
    if length == 0 {
        return data
    }
    
    unpadding := int(data[length-1])
    if unpadding > length {
        return data
    }
    
    return data[:(length - unpadding)]
}
```

## Asymmetric Encryption

### RSA Implementation

```go
package main

import (
    "crypto/rand"
    "crypto/rsa"
    "crypto/sha256"
    "crypto/x509"
    "encoding/pem"
    "fmt"
    "io"
)

type RSAEncryption struct {
    privateKey *rsa.PrivateKey
    publicKey  *rsa.PublicKey
}

func NewRSAEncryption(bits int) (*RSAEncryption, error) {
    privateKey, err := rsa.GenerateKey(rand.Reader, bits)
    if err != nil {
        return nil, err
    }
    
    return &RSAEncryption{
        privateKey: privateKey,
        publicKey:  &privateKey.PublicKey,
    }, nil
}

func (rsa *RSAEncryption) Encrypt(plaintext []byte) ([]byte, error) {
    return rsa.EncryptWithPublicKey(plaintext, rsa.publicKey)
}

func (rsa *RSAEncryption) EncryptWithPublicKey(plaintext []byte, publicKey *rsa.PublicKey) ([]byte, error) {
    return rsa.EncryptOAEP(plaintext, publicKey)
}

func (rsa *RSAEncryption) EncryptOAEP(plaintext []byte, publicKey *rsa.PublicKey) ([]byte, error) {
    hash := sha256.New()
    return rsa.EncryptOAEP(hash, rand.Reader, publicKey, plaintext, nil)
}

func (rsa *RSAEncryption) Decrypt(ciphertext []byte) ([]byte, error) {
    return rsa.DecryptOAEP(ciphertext)
}

func (rsa *RSAEncryption) DecryptOAEP(ciphertext []byte) ([]byte, error) {
    hash := sha256.New()
    return rsa.DecryptOAEP(hash, rand.Reader, rsa.privateKey, ciphertext, nil)
}

func (rsa *RSAEncryption) Sign(message []byte) ([]byte, error) {
    hash := sha256.Sum256(message)
    return rsa.SignPKCS1v15(rand.Reader, rsa.privateKey, crypto.SHA256, hash[:])
}

func (rsa *RSAEncryption) Verify(message []byte, signature []byte) error {
    hash := sha256.Sum256(message)
    return rsa.VerifyPKCS1v15(rsa.publicKey, crypto.SHA256, hash[:], signature)
}

// Key Serialization
func (rsa *RSAEncryption) ExportPrivateKey() ([]byte, error) {
    privateKeyBytes := x509.MarshalPKCS1PrivateKey(rsa.privateKey)
    privateKeyPEM := pem.EncodeToMemory(&pem.Block{
        Type:  "RSA PRIVATE KEY",
        Bytes: privateKeyBytes,
    })
    return privateKeyPEM, nil
}

func (rsa *RSAEncryption) ExportPublicKey() ([]byte, error) {
    publicKeyBytes := x509.MarshalPKCS1PublicKey(rsa.publicKey)
    publicKeyPEM := pem.EncodeToMemory(&pem.Block{
        Type:  "RSA PUBLIC KEY",
        Bytes: publicKeyBytes,
    })
    return publicKeyPEM, nil
}

func LoadRSAFromPEM(privateKeyPEM []byte) (*RSAEncryption, error) {
    block, _ := pem.Decode(privateKeyPEM)
    if block == nil {
        return nil, fmt.Errorf("failed to decode PEM block")
    }
    
    privateKey, err := x509.ParsePKCS1PrivateKey(block.Bytes)
    if err != nil {
        return nil, err
    }
    
    return &RSAEncryption{
        privateKey: privateKey,
        publicKey:  &privateKey.PublicKey,
    }, nil
}
```

## Hash Functions

### Cryptographic Hash Implementation

```go
package main

import (
    "crypto/md5"
    "crypto/sha1"
    "crypto/sha256"
    "crypto/sha512"
    "encoding/hex"
    "fmt"
    "hash"
)

type HashFunction struct {
    algorithm string
    hasher    hash.Hash
}

func NewHashFunction(algorithm string) (*HashFunction, error) {
    var hasher hash.Hash
    
    switch algorithm {
    case "md5":
        hasher = md5.New()
    case "sha1":
        hasher = sha1.New()
    case "sha256":
        hasher = sha256.New()
    case "sha512":
        hasher = sha512.New()
    default:
        return nil, fmt.Errorf("unsupported algorithm: %s", algorithm)
    }
    
    return &HashFunction{
        algorithm: algorithm,
        hasher:    hasher,
    }, nil
}

func (hf *HashFunction) Hash(data []byte) string {
    hf.hasher.Reset()
    hf.hasher.Write(data)
    return hex.EncodeToString(hf.hasher.Sum(nil))
}

func (hf *HashFunction) HashWithSalt(data []byte, salt []byte) string {
    hf.hasher.Reset()
    hf.hasher.Write(data)
    hf.hasher.Write(salt)
    return hex.EncodeToString(hf.hasher.Sum(nil))
}

// Password Hashing with Salt
type PasswordHasher struct {
    saltLength int
}

func NewPasswordHasher(saltLength int) *PasswordHasher {
    return &PasswordHasher{saltLength: saltLength}
}

func (ph *PasswordHasher) HashPassword(password string) (string, string, error) {
    salt := make([]byte, ph.saltLength)
    if _, err := rand.Read(salt); err != nil {
        return "", "", err
    }
    
    hasher := sha256.New()
    hasher.Write([]byte(password))
    hasher.Write(salt)
    hash := hasher.Sum(nil)
    
    return hex.EncodeToString(hash), hex.EncodeToString(salt), nil
}

func (ph *PasswordHasher) VerifyPassword(password, hash, salt string) bool {
    saltBytes, err := hex.DecodeString(salt)
    if err != nil {
        return false
    }
    
    hasher := sha256.New()
    hasher.Write([]byte(password))
    hasher.Write(saltBytes)
    computedHash := hasher.Sum(nil)
    
    expectedHash, err := hex.DecodeString(hash)
    if err != nil {
        return false
    }
    
    return bytes.Equal(computedHash, expectedHash)
}

// HMAC Implementation
func ComputeHMAC(data []byte, key []byte) string {
    h := hmac.New(sha256.New, key)
    h.Write(data)
    return hex.EncodeToString(h.Sum(nil))
}

func VerifyHMAC(data []byte, key []byte, expectedMAC string) bool {
    computedMAC := ComputeHMAC(data, key)
    return hmac.Equal([]byte(computedMAC), []byte(expectedMAC))
}
```

## Digital Signatures

### ECDSA Implementation

```go
package main

import (
    "crypto/ecdsa"
    "crypto/elliptic"
    "crypto/rand"
    "crypto/sha256"
    "encoding/hex"
    "fmt"
    "math/big"
)

type ECDSASigner struct {
    privateKey *ecdsa.PrivateKey
    publicKey  *ecdsa.PublicKey
}

func NewECDSASigner() (*ECDSASigner, error) {
    privateKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
    if err != nil {
        return nil, err
    }
    
    return &ECDSASigner{
        privateKey: privateKey,
        publicKey:  &privateKey.PublicKey,
    }, nil
}

func (ecdsa *ECDSASigner) Sign(message []byte) ([]byte, error) {
    hash := sha256.Sum256(message)
    r, s, err := ecdsa.Sign(rand.Reader, ecdsa.privateKey, hash[:])
    if err != nil {
        return nil, err
    }
    
    // Encode signature as r||s
    signature := append(r.Bytes(), s.Bytes()...)
    return signature, nil
}

func (ecdsa *ECDSASigner) Verify(message []byte, signature []byte) bool {
    hash := sha256.Sum256(message)
    
    // Decode signature
    keySize := ecdsa.privateKey.Curve.Params().BitSize / 8
    if len(signature) != 2*keySize {
        return false
    }
    
    r := new(big.Int).SetBytes(signature[:keySize])
    s := new(big.Int).SetBytes(signature[keySize:])
    
    return ecdsa.Verify(ecdsa.publicKey, hash[:], r, s)
}

// Message Authentication Code
type MACGenerator struct {
    key []byte
}

func NewMACGenerator(key []byte) *MACGenerator {
    return &MACGenerator{key: key}
}

func (mac *MACGenerator) GenerateMAC(message []byte) string {
    h := hmac.New(sha256.New, mac.key)
    h.Write(message)
    return hex.EncodeToString(h.Sum(nil))
}

func (mac *MACGenerator) VerifyMAC(message []byte, expectedMAC string) bool {
    computedMAC := mac.GenerateMAC(message)
    return hmac.Equal([]byte(computedMAC), []byte(expectedMAC))
}
```

## Key Management

### Key Derivation and Storage

```go
package main

import (
    "crypto/rand"
    "crypto/sha256"
    "encoding/hex"
    "fmt"
    "golang.org/x/crypto/pbkdf2"
    "golang.org/x/crypto/scrypt"
)

type KeyManager struct {
    masterKey []byte
}

func NewKeyManager(masterKey []byte) *KeyManager {
    return &KeyManager{masterKey: masterKey}
}

// PBKDF2 Key Derivation
func (km *KeyManager) DeriveKeyPBKDF2(password []byte, salt []byte, iterations int, keyLength int) []byte {
    return pbkdf2.Key(password, salt, iterations, keyLength, sha256.New)
}

// Scrypt Key Derivation
func (km *KeyManager) DeriveKeyScrypt(password []byte, salt []byte, N, r, p, keyLength int) ([]byte, error) {
    return scrypt.Key(password, salt, N, r, p, keyLength)
}

// Key Rotation
func (km *KeyManager) RotateKey() ([]byte, error) {
    newKey := make([]byte, 32)
    if _, err := rand.Read(newKey); err != nil {
        return nil, err
    }
    
    km.masterKey = newKey
    return newKey, nil
}

// Key Encryption
func (km *KeyManager) EncryptKey(key []byte) ([]byte, error) {
    // Use master key to encrypt other keys
    block, err := aes.NewCipher(km.masterKey)
    if err != nil {
        return nil, err
    }
    
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }
    
    nonce := make([]byte, gcm.NonceSize())
    if _, err = rand.Read(nonce); err != nil {
        return nil, err
    }
    
    ciphertext := gcm.Seal(nonce, nonce, key, nil)
    return ciphertext, nil
}

func (km *KeyManager) DecryptKey(encryptedKey []byte) ([]byte, error) {
    block, err := aes.NewCipher(km.masterKey)
    if err != nil {
        return nil, err
    }
    
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }
    
    nonceSize := gcm.NonceSize()
    if len(encryptedKey) < nonceSize {
        return nil, fmt.Errorf("ciphertext too short")
    }
    
    nonce, ciphertext := encryptedKey[:nonceSize], encryptedKey[nonceSize:]
    key, err := gcm.Open(nil, nonce, ciphertext, nil)
    if err != nil {
        return nil, err
    }
    
    return key, nil
}

// Secure Key Storage
type SecureKeyStore struct {
    keys map[string][]byte
    km   *KeyManager
}

func NewSecureKeyStore(masterKey []byte) *SecureKeyStore {
    return &SecureKeyStore{
        keys: make(map[string][]byte),
        km:   NewKeyManager(masterKey),
    }
}

func (sks *SecureKeyStore) StoreKey(keyID string, key []byte) error {
    encryptedKey, err := sks.km.EncryptKey(key)
    if err != nil {
        return err
    }
    
    sks.keys[keyID] = encryptedKey
    return nil
}

func (sks *SecureKeyStore) RetrieveKey(keyID string) ([]byte, error) {
    encryptedKey, exists := sks.keys[keyID]
    if !exists {
        return nil, fmt.Errorf("key not found")
    }
    
    return sks.km.DecryptKey(encryptedKey)
}

func (sks *SecureKeyStore) DeleteKey(keyID string) {
    delete(sks.keys, keyID)
}
```

## TLS/SSL Implementation

### Custom TLS Server

```go
package main

import (
    "crypto/tls"
    "crypto/x509"
    "fmt"
    "io"
    "net"
    "net/http"
    "time"
)

type TLSServer struct {
    config *tls.Config
    server *http.Server
}

func NewTLSServer(certFile, keyFile string) (*TLSServer, error) {
    // Load certificate and key
    cert, err := tls.LoadX509KeyPair(certFile, keyFile)
    if err != nil {
        return nil, err
    }
    
    config := &tls.Config{
        Certificates: []tls.Certificate{cert},
        MinVersion:   tls.VersionTLS12,
        CipherSuites: []uint16{
            tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
            tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
            tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
        },
    }
    
    return &TLSServer{config: config}, nil
}

func (ts *TLSServer) Start(addr string) error {
    listener, err := tls.Listen("tcp", addr, ts.config)
    if err != nil {
        return err
    }
    
    ts.server = &http.Server{
        Addr:      addr,
        TLSConfig: ts.config,
        Handler:   ts.createHandler(),
    }
    
    go func() {
        if err := ts.server.Serve(listener); err != nil && err != http.ErrServerClosed {
            fmt.Printf("TLS server error: %v\n", err)
        }
    }()
    
    return nil
}

func (ts *TLSServer) createHandler() http.Handler {
    mux := http.NewServeMux()
    
    mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        fmt.Fprintf(w, "Secure connection established!\n")
    })
    
    mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        fmt.Fprintf(w, "Server is healthy\n")
    })
    
    return mux
}

// TLS Client
type TLSClient struct {
    config *tls.Config
    client *http.Client
}

func NewTLSClient() *TLSClient {
    config := &tls.Config{
        InsecureSkipVerify: false, // Set to true only for testing
        MinVersion:         tls.VersionTLS12,
    }
    
    client := &http.Client{
        Transport: &http.Transport{
            TLSClientConfig: config,
        },
        Timeout: 30 * time.Second,
    }
    
    return &TLSClient{
        config: config,
        client: client,
    }
}

func (tc *TLSClient) Get(url string) (*http.Response, error) {
    return tc.client.Get(url)
}

func (tc *TLSClient) Post(url, contentType string, body io.Reader) (*http.Response, error) {
    return tc.client.Post(url, contentType, body)
}

// Certificate Validation
func ValidateCertificate(certPEM []byte) error {
    block, _ := pem.Decode(certPEM)
    if block == nil {
        return fmt.Errorf("failed to decode PEM block")
    }
    
    cert, err := x509.ParseCertificate(block.Bytes)
    if err != nil {
        return err
    }
    
    // Check if certificate is expired
    if time.Now().After(cert.NotAfter) {
        return fmt.Errorf("certificate has expired")
    }
    
    // Check if certificate is not yet valid
    if time.Now().Before(cert.NotBefore) {
        return fmt.Errorf("certificate is not yet valid")
    }
    
    return nil
}
```

## OAuth 2.0 & JWT

### JWT Implementation

```go
package main

import (
    "crypto/hmac"
    "crypto/sha256"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "strings"
    "time"
)

type JWTClaims struct {
    UserID    string `json:"user_id"`
    Username  string `json:"username"`
    Role      string `json:"role"`
    ExpiresAt int64  `json:"exp"`
    IssuedAt  int64  `json:"iat"`
    Issuer    string `json:"iss"`
}

type JWTManager struct {
    secretKey []byte
}

func NewJWTManager(secretKey string) *JWTManager {
    return &JWTManager{secretKey: []byte(secretKey)}
}

func (jm *JWTManager) GenerateToken(claims JWTClaims) (string, error) {
    // Set standard claims
    now := time.Now().Unix()
    claims.IssuedAt = now
    claims.ExpiresAt = now + 3600 // 1 hour
    
    // Create header
    header := map[string]string{
        "alg": "HS256",
        "typ": "JWT",
    }
    
    headerJSON, err := json.Marshal(header)
    if err != nil {
        return "", err
    }
    
    claimsJSON, err := json.Marshal(claims)
    if err != nil {
        return "", err
    }
    
    // Encode header and claims
    headerEncoded := base64.RawURLEncoding.EncodeToString(headerJSON)
    claimsEncoded := base64.RawURLEncoding.EncodeToString(claimsJSON)
    
    // Create signature
    message := headerEncoded + "." + claimsEncoded
    signature := jm.createSignature(message)
    
    return message + "." + signature, nil
}

func (jm *JWTManager) ValidateToken(token string) (*JWTClaims, error) {
    parts := strings.Split(token, ".")
    if len(parts) != 3 {
        return nil, fmt.Errorf("invalid token format")
    }
    
    // Verify signature
    message := parts[0] + "." + parts[1]
    expectedSignature := jm.createSignature(message)
    if parts[2] != expectedSignature {
        return nil, fmt.Errorf("invalid signature")
    }
    
    // Decode claims
    claimsJSON, err := base64.RawURLEncoding.DecodeString(parts[1])
    if err != nil {
        return nil, err
    }
    
    var claims JWTClaims
    if err := json.Unmarshal(claimsJSON, &claims); err != nil {
        return nil, err
    }
    
    // Check expiration
    if time.Now().Unix() > claims.ExpiresAt {
        return nil, fmt.Errorf("token has expired")
    }
    
    return &claims, nil
}

func (jm *JWTManager) createSignature(message string) string {
    h := hmac.New(sha256.New, jm.secretKey)
    h.Write([]byte(message))
    return base64.RawURLEncoding.EncodeToString(h.Sum(nil))
}

// OAuth 2.0 Authorization Server
type OAuth2Server struct {
    clients map[string]*OAuth2Client
    tokens  map[string]*AccessToken
}

type OAuth2Client struct {
    ID          string
    Secret      string
    RedirectURI string
    Scopes      []string
}

type AccessToken struct {
    Token     string
    ClientID  string
    UserID    string
    Scopes    []string
    ExpiresAt time.Time
}

func NewOAuth2Server() *OAuth2Server {
    return &OAuth2Server{
        clients: make(map[string]*OAuth2Client),
        tokens:  make(map[string]*AccessToken),
    }
}

func (os *OAuth2Server) RegisterClient(client *OAuth2Client) {
    os.clients[client.ID] = client
}

func (os *OAuth2Server) Authorize(clientID, redirectURI, scope string) (string, error) {
    client, exists := os.clients[clientID]
    if !exists {
        return "", fmt.Errorf("invalid client")
    }
    
    if client.RedirectURI != redirectURI {
        return "", fmt.Errorf("invalid redirect URI")
    }
    
    // Generate authorization code
    authCode := generateRandomString(32)
    
    // In real implementation, store auth code with expiration
    return authCode, nil
}

func (os *OAuth2Server) ExchangeCode(clientID, clientSecret, authCode string) (*AccessToken, error) {
    client, exists := os.clients[clientID]
    if !exists || client.Secret != clientSecret {
        return nil, fmt.Errorf("invalid client credentials")
    }
    
    // In real implementation, validate auth code
    if authCode == "" {
        return nil, fmt.Errorf("invalid authorization code")
    }
    
    // Generate access token
    token := &AccessToken{
        Token:     generateRandomString(64),
        ClientID:  clientID,
        UserID:    "user123", // In real implementation, get from auth code
        Scopes:    []string{"read", "write"},
        ExpiresAt: time.Now().Add(1 * time.Hour),
    }
    
    os.tokens[token.Token] = token
    return token, nil
}

func (os *OAuth2Server) ValidateToken(token string) (*AccessToken, error) {
    accessToken, exists := os.tokens[token]
    if !exists {
        return nil, fmt.Errorf("invalid token")
    }
    
    if time.Now().After(accessToken.ExpiresAt) {
        delete(os.tokens, token)
        return nil, fmt.Errorf("token has expired")
    }
    
    return accessToken, nil
}

func generateRandomString(length int) string {
    const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    b := make([]byte, length)
    for i := range b {
        b[i] = charset[rand.Intn(len(charset))]
    }
    return string(b)
}
```

## Security Best Practices

### Input Validation and Sanitization

```go
package main

import (
    "fmt"
    "net/url"
    "regexp"
    "strings"
    "unicode"
)

type InputValidator struct {
    emailRegex    *regexp.Regexp
    phoneRegex    *regexp.Regexp
    usernameRegex *regexp.Regexp
}

func NewInputValidator() *InputValidator {
    return &InputValidator{
        emailRegex:    regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`),
        phoneRegex:    regexp.MustCompile(`^\+?[1-9]\d{1,14}$`),
        usernameRegex: regexp.MustCompile(`^[a-zA-Z0-9_-]{3,20}$`),
    }
}

func (iv *InputValidator) ValidateEmail(email string) error {
    if !iv.emailRegex.MatchString(email) {
        return fmt.Errorf("invalid email format")
    }
    
    if len(email) > 254 {
        return fmt.Errorf("email too long")
    }
    
    return nil
}

func (iv *InputValidator) ValidatePhone(phone string) error {
    if !iv.phoneRegex.MatchString(phone) {
        return fmt.Errorf("invalid phone number format")
    }
    
    return nil
}

func (iv *InputValidator) ValidateUsername(username string) error {
    if !iv.usernameRegex.MatchString(username) {
        return fmt.Errorf("invalid username format")
    }
    
    return nil
}

func (iv *InputValidator) SanitizeInput(input string) string {
    // Remove null bytes
    input = strings.ReplaceAll(input, "\x00", "")
    
    // Trim whitespace
    input = strings.TrimSpace(input)
    
    // Remove control characters
    var result strings.Builder
    for _, r := range input {
        if unicode.IsPrint(r) || unicode.IsSpace(r) {
            result.WriteRune(r)
        }
    }
    
    return result.String()
}

func (iv *InputValidator) ValidateURL(rawURL string) error {
    parsedURL, err := url.Parse(rawURL)
    if err != nil {
        return fmt.Errorf("invalid URL format")
    }
    
    if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
        return fmt.Errorf("unsupported URL scheme")
    }
    
    if len(parsedURL.Host) == 0 {
        return fmt.Errorf("missing host in URL")
    }
    
    return nil
}

// Rate Limiting
type RateLimiter struct {
    requests map[string][]time.Time
    limit    int
    window   time.Duration
}

func NewRateLimiter(limit int, window time.Duration) *RateLimiter {
    return &RateLimiter{
        requests: make(map[string][]time.Time),
        limit:    limit,
        window:   window,
    }
}

func (rl *RateLimiter) IsAllowed(clientID string) bool {
    now := time.Now()
    
    // Clean old requests
    if requests, exists := rl.requests[clientID]; exists {
        var validRequests []time.Time
        for _, reqTime := range requests {
            if now.Sub(reqTime) < rl.window {
                validRequests = append(validRequests, reqTime)
            }
        }
        rl.requests[clientID] = validRequests
    }
    
    // Check if limit exceeded
    if len(rl.requests[clientID]) >= rl.limit {
        return false
    }
    
    // Add current request
    rl.requests[clientID] = append(rl.requests[clientID], now)
    return true
}

// SQL Injection Prevention
func SanitizeSQLInput(input string) string {
    // Remove SQL injection patterns
    dangerous := []string{
        "'", "\"", ";", "--", "/*", "*/", "xp_", "sp_",
        "UNION", "SELECT", "INSERT", "UPDATE", "DELETE",
        "DROP", "CREATE", "ALTER", "EXEC", "EXECUTE",
    }
    
    result := input
    for _, pattern := range dangerous {
        result = strings.ReplaceAll(strings.ToUpper(result), strings.ToUpper(pattern), "")
    }
    
    return result
}

// XSS Prevention
func SanitizeHTML(input string) string {
    // Remove HTML tags
    htmlRegex := regexp.MustCompile(`<[^>]*>`)
    result := htmlRegex.ReplaceAllString(input, "")
    
    // Escape special characters
    result = strings.ReplaceAll(result, "&", "&amp;")
    result = strings.ReplaceAll(result, "<", "&lt;")
    result = strings.ReplaceAll(result, ">", "&gt;")
    result = strings.ReplaceAll(result, "\"", "&quot;")
    result = strings.ReplaceAll(result, "'", "&#x27;")
    
    return result
}
```

## Interview Questions

### Basic Concepts
1. **What is the difference between encryption, hashing, and encoding?**
2. **Explain symmetric vs asymmetric encryption.**
3. **What are the advantages and disadvantages of RSA vs ECDSA?**
4. **How does TLS/SSL work?**
5. **What is the purpose of digital signatures?**

### Advanced Topics
1. **How would you implement a secure key management system?**
2. **Explain the OAuth 2.0 flow.**
3. **How do you prevent common security vulnerabilities?**
4. **What are the security implications of different hash functions?**
5. **How would you implement certificate pinning?**

### System Design
1. **Design a secure authentication system.**
2. **How would you implement end-to-end encryption?**
3. **Design a secure API gateway.**
4. **How would you implement secure communication between microservices?**
5. **Design a secure data storage system.**

## Conclusion

Advanced security and cryptography are essential for building secure systems. Key areas to master:

- **Cryptographic Fundamentals**: Encryption, hashing, digital signatures
- **Key Management**: Generation, storage, rotation, distribution
- **TLS/SSL**: Secure communication protocols
- **OAuth 2.0 & JWT**: Authentication and authorization
- **Security Best Practices**: Input validation, rate limiting, vulnerability prevention
- **Compliance**: Security standards and regulations

Understanding these concepts helps in:
- Building secure applications
- Implementing proper authentication
- Protecting sensitive data
- Preventing security vulnerabilities
- Preparing for technical interviews

This guide provides a comprehensive foundation for advanced security concepts and their practical implementation in Go.
