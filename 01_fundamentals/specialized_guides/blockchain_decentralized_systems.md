# Blockchain and Decentralized Systems Guide

## Table of Contents
- [Introduction](#introduction)
- [Blockchain Fundamentals](#blockchain-fundamentals)
- [Cryptocurrency Systems](#cryptocurrency-systems)
- [Smart Contracts](#smart-contracts)
- [Decentralized Applications (DApps)](#decentralized-applications-dapps)
- [Consensus Mechanisms](#consensus-mechanisms)
- [DeFi (Decentralized Finance)](#defi-decentralized-finance)
- [NFTs and Digital Assets](#nfts-and-digital-assets)
- [Web3 Infrastructure](#web3-infrastructure)
- [Security and Privacy](#security-and-privacy)

## Introduction

Blockchain and decentralized systems represent a paradigm shift in how we build and interact with digital systems. This guide covers the essential concepts, technologies, and implementation patterns for building blockchain-based applications.

## Blockchain Fundamentals

### Blockchain Data Structure

```go
// Blockchain Core Components
type Block struct {
    Index        int64
    Timestamp    time.Time
    Data         []Transaction
    PreviousHash string
    Hash         string
    Nonce        int64
    MerkleRoot   string
}

type Transaction struct {
    ID        string
    From      string
    To        string
    Amount    decimal.Decimal
    GasPrice  decimal.Decimal
    GasLimit  int64
    Nonce     int64
    Signature string
    Timestamp time.Time
}

type Blockchain struct {
    blocks    []*Block
    difficulty int
    mu        sync.RWMutex
}

func NewBlockchain(difficulty int) *Blockchain {
    genesisBlock := &Block{
        Index:        0,
        Timestamp:    time.Now(),
        Data:         []Transaction{},
        PreviousHash: "0",
        Nonce:        0,
    }
    
    genesisBlock.Hash = calculateHash(genesisBlock)
    
    return &Blockchain{
        blocks:     []*Block{genesisBlock},
        difficulty: difficulty,
    }
}

func (bc *Blockchain) AddBlock(transactions []Transaction) *Block {
    bc.mu.Lock()
    defer bc.mu.Unlock()
    
    previousBlock := bc.blocks[len(bc.blocks)-1]
    newBlock := &Block{
        Index:        previousBlock.Index + 1,
        Timestamp:    time.Now(),
        Data:         transactions,
        PreviousHash: previousBlock.Hash,
        Nonce:        0,
    }
    
    // Calculate Merkle root
    newBlock.MerkleRoot = calculateMerkleRoot(transactions)
    
    // Mine block
    newBlock = bc.mineBlock(newBlock)
    
    bc.blocks = append(bc.blocks, newBlock)
    return newBlock
}

func (bc *Blockchain) mineBlock(block *Block) *Block {
    target := strings.Repeat("0", bc.difficulty)
    
    for {
        block.Hash = calculateHash(block)
        if strings.HasPrefix(block.Hash, target) {
            break
        }
        block.Nonce++
    }
    
    return block
}

func calculateHash(block *Block) string {
    data := fmt.Sprintf("%d%s%s%d%s", 
        block.Index, 
        block.Timestamp.Format(time.RFC3339), 
        block.PreviousHash, 
        block.Nonce, 
        block.MerkleRoot)
    
    hash := sha256.Sum256([]byte(data))
    return hex.EncodeToString(hash[:])
}
```

### Merkle Tree Implementation

```go
// Merkle Tree for efficient transaction verification
type MerkleTree struct {
    root   *MerkleNode
    leaves []*MerkleNode
}

type MerkleNode struct {
    left   *MerkleNode
    right  *MerkleNode
    data   []byte
    hash   string
}

func NewMerkleTree(data [][]byte) *MerkleTree {
    if len(data) == 0 {
        return nil
    }
    
    var leaves []*MerkleNode
    for _, d := range data {
        leaves = append(leaves, &MerkleNode{
            data: d,
            hash: calculateHash(d),
        })
    }
    
    root := buildMerkleTree(leaves)
    
    return &MerkleTree{
        root:   root,
        leaves: leaves,
    }
}

func buildMerkleTree(nodes []*MerkleNode) *MerkleNode {
    if len(nodes) == 1 {
        return nodes[0]
    }
    
    var nextLevel []*MerkleNode
    
    for i := 0; i < len(nodes); i += 2 {
        left := nodes[i]
        right := nodes[i+1]
        
        if right == nil {
            right = left
        }
        
        combined := append(left.hash, right.hash...)
        parent := &MerkleNode{
            left:  left,
            right: right,
            hash:  calculateHash(combined),
        }
        
        nextLevel = append(nextLevel, parent)
    }
    
    return buildMerkleTree(nextLevel)
}

func (mt *MerkleTree) GetRootHash() string {
    return mt.root.hash
}

func (mt *MerkleTree) VerifyProof(leafData []byte, proof []string) bool {
    leafHash := calculateHash(leafData)
    currentHash := leafHash
    
    for _, proofHash := range proof {
        combined := append(currentHash, proofHash...)
        currentHash = calculateHash(combined)
    }
    
    return currentHash == mt.root.hash
}
```

## Cryptocurrency Systems

### Wallet Implementation

```go
// Cryptocurrency Wallet
type Wallet struct {
    privateKey *ecdsa.PrivateKey
    publicKey  *ecdsa.PublicKey
    address    string
}

func NewWallet() *Wallet {
    privateKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
    if err != nil {
        panic(err)
    }
    
    publicKey := &privateKey.PublicKey
    address := generateAddress(publicKey)
    
    return &Wallet{
        privateKey: privateKey,
        publicKey:  publicKey,
        address:    address,
    }
}

func (w *Wallet) SignTransaction(tx *Transaction) (string, error) {
    txHash := tx.CalculateHash()
    
    r, s, err := ecdsa.Sign(rand.Reader, w.privateKey, txHash)
    if err != nil {
        return "", err
    }
    
    signature := append(r.Bytes(), s.Bytes()...)
    return hex.EncodeToString(signature), nil
}

func (w *Wallet) VerifySignature(tx *Transaction, signature string) bool {
    sigBytes, err := hex.DecodeString(signature)
    if err != nil {
        return false
    }
    
    if len(sigBytes) != 64 {
        return false
    }
    
    r := new(big.Int).SetBytes(sigBytes[:32])
    s := new(big.Int).SetBytes(sigBytes[32:])
    
    txHash := tx.CalculateHash()
    
    return ecdsa.Verify(w.publicKey, txHash, r, s)
}

func generateAddress(publicKey *ecdsa.PublicKey) string {
    publicKeyBytes := append(publicKey.X.Bytes(), publicKey.Y.Bytes()...)
    hash := sha256.Sum256(publicKeyBytes)
    return hex.EncodeToString(hash[:20])
}
```

### Transaction Pool

```go
// Transaction Pool for managing pending transactions
type TransactionPool struct {
    transactions map[string]*Transaction
    mu           sync.RWMutex
    maxSize      int
}

func NewTransactionPool(maxSize int) *TransactionPool {
    return &TransactionPool{
        transactions: make(map[string]*Transaction),
        maxSize:      maxSize,
    }
}

func (tp *TransactionPool) AddTransaction(tx *Transaction) error {
    tp.mu.Lock()
    defer tp.mu.Unlock()
    
    if len(tp.transactions) >= tp.maxSize {
        return fmt.Errorf("transaction pool is full")
    }
    
    if !tp.validateTransaction(tx) {
        return fmt.Errorf("invalid transaction")
    }
    
    tp.transactions[tx.ID] = tx
    return nil
}

func (tp *TransactionPool) GetTransactions(limit int) []*Transaction {
    tp.mu.RLock()
    defer tp.mu.RUnlock()
    
    var transactions []*Transaction
    count := 0
    
    for _, tx := range tp.transactions {
        if count >= limit {
            break
        }
        transactions = append(transactions, tx)
        count++
    }
    
    return transactions
}

func (tp *TransactionPool) RemoveTransactions(txIDs []string) {
    tp.mu.Lock()
    defer tp.mu.Unlock()
    
    for _, txID := range txIDs {
        delete(tp.transactions, txID)
    }
}

func (tp *TransactionPool) validateTransaction(tx *Transaction) bool {
    // Validate transaction structure
    if tx.From == "" || tx.To == "" || tx.Amount.LessThanOrEqual(decimal.Zero) {
        return false
    }
    
    // Validate signature
    // Implementation depends on wallet system
    
    return true
}
```

## Smart Contracts

### Smart Contract Engine

```go
// Smart Contract Virtual Machine
type SmartContractVM struct {
    contracts map[string]*SmartContract
    state     map[string]interface{}
    gasLimit  int64
    mu        sync.RWMutex
}

type SmartContract struct {
    Address string
    Code    []byte
    ABI     []byte
    State   map[string]interface{}
}

type ContractCall struct {
    ContractAddress string
    Function        string
    Parameters      []interface{}
    GasLimit        int64
    Value           decimal.Decimal
}

func NewSmartContractVM() *SmartContractVM {
    return &SmartContractVM{
        contracts: make(map[string]*SmartContract),
        state:     make(map[string]interface{}),
        gasLimit:  1000000,
    }
}

func (vm *SmartContractVM) DeployContract(code []byte, abi []byte) (string, error) {
    vm.mu.Lock()
    defer vm.mu.Unlock()
    
    address := generateContractAddress(code)
    
    contract := &SmartContract{
        Address: address,
        Code:    code,
        ABI:     abi,
        State:   make(map[string]interface{}),
    }
    
    vm.contracts[address] = contract
    return address, nil
}

func (vm *SmartContractVM) CallContract(call *ContractCall) (interface{}, error) {
    vm.mu.RLock()
    contract := vm.contracts[call.ContractAddress]
    vm.mu.RUnlock()
    
    if contract == nil {
        return nil, fmt.Errorf("contract not found")
    }
    
    // Execute contract function
    result, gasUsed, err := vm.executeFunction(contract, call)
    if err != nil {
        return nil, err
    }
    
    if gasUsed > call.GasLimit {
        return nil, fmt.Errorf("gas limit exceeded")
    }
    
    return result, nil
}

func (vm *SmartContractVM) executeFunction(contract *SmartContract, call *ContractCall) (interface{}, int64, error) {
    // Simplified execution - in reality, this would be a full VM
    // This is a placeholder for actual smart contract execution
    
    switch call.Function {
    case "transfer":
        return vm.executeTransfer(contract, call.Parameters)
    case "balanceOf":
        return vm.executeBalanceOf(contract, call.Parameters)
    default:
        return nil, 0, fmt.Errorf("unknown function: %s", call.Function)
    }
}

func (vm *SmartContractVM) executeTransfer(contract *SmartContract, params []interface{}) (interface{}, int64, error) {
    if len(params) != 2 {
        return nil, 0, fmt.Errorf("invalid parameters for transfer")
    }
    
    to := params[0].(string)
    amount := params[1].(decimal.Decimal)
    
    // Update contract state
    contract.State["balance_"+to] = amount
    
    return true, 21000, nil // Standard gas cost for transfer
}

func (vm *SmartContractVM) executeBalanceOf(contract *SmartContract, params []interface{}) (interface{}, int64, error) {
    if len(params) != 1 {
        return nil, 0, fmt.Errorf("invalid parameters for balanceOf")
    }
    
    address := params[0].(string)
    balance := contract.State["balance_"+address]
    
    if balance == nil {
        return decimal.Zero, 2000, nil
    }
    
    return balance, 2000, nil
}
```

## Decentralized Applications (DApps)

### DApp Frontend Integration

```go
// Web3 Integration for DApps
type Web3Client struct {
    provider    string
    chainID     int64
    accounts    []string
    contracts   map[string]*Contract
    mu          sync.RWMutex
}

type Contract struct {
    Address string
    ABI     []byte
    Client  *Web3Client
}

func NewWeb3Client(provider string, chainID int64) *Web3Client {
    return &Web3Client{
        provider:  provider,
        chainID:   chainID,
        accounts:  make([]string, 0),
        contracts: make(map[string]*Contract),
    }
}

func (w3 *Web3Client) Connect() error {
    // Connect to Web3 provider (MetaMask, etc.)
    // This is a simplified implementation
    
    // Get accounts
    accounts, err := w3.getAccounts()
    if err != nil {
        return err
    }
    
    w3.mu.Lock()
    w3.accounts = accounts
    w3.mu.Unlock()
    
    return nil
}

func (w3 *Web3Client) LoadContract(address string, abi []byte) (*Contract, error) {
    w3.mu.Lock()
    defer w3.mu.Unlock()
    
    contract := &Contract{
        Address: address,
        ABI:     abi,
        Client:  w3,
    }
    
    w3.contracts[address] = contract
    return contract, nil
}

func (c *Contract) Call(function string, params []interface{}) (interface{}, error) {
    // Call contract function
    call := &ContractCall{
        ContractAddress: c.Address,
        Function:        function,
        Parameters:      params,
        GasLimit:        100000,
        Value:           decimal.Zero,
    }
    
    return c.Client.callContract(call)
}

func (c *Contract) Send(function string, params []interface{}, value decimal.Decimal) (string, error) {
    // Send transaction to contract
    tx := &Transaction{
        To:      c.Address,
        Value:   value,
        Data:    c.encodeFunctionCall(function, params),
        GasLimit: 100000,
    }
    
    return c.Client.sendTransaction(tx)
}

func (c *Contract) encodeFunctionCall(function string, params []interface{}) []byte {
    // Encode function call data
    // This is a simplified implementation
    // In reality, you'd use proper ABI encoding
    
    data := []byte(function)
    for _, param := range params {
        data = append(data, []byte(fmt.Sprintf("%v", param))...)
    }
    
    return data
}
```

## Consensus Mechanisms

### Proof of Work (PoW)

```go
// Proof of Work Implementation
type ProofOfWork struct {
    difficulty int
    target     *big.Int
}

func NewProofOfWork(difficulty int) *ProofOfWork {
    target := big.NewInt(1)
    target.Lsh(target, uint(256-difficulty))
    
    return &ProofOfWork{
        difficulty: difficulty,
        target:     target,
    }
}

func (pow *ProofOfWork) MineBlock(block *Block) *Block {
    var hashInt big.Int
    nonce := 0
    
    for {
        block.Nonce = int64(nonce)
        hash := calculateHash(block)
        
        hashInt.SetString(hash, 16)
        if hashInt.Cmp(pow.target) == -1 {
            block.Hash = hash
            break
        }
        
        nonce++
    }
    
    return block
}

func (pow *ProofOfWork) ValidateBlock(block *Block) bool {
    var hashInt big.Int
    hash := calculateHash(block)
    hashInt.SetString(hash, 16)
    
    return hashInt.Cmp(pow.target) == -1
}
```

### Proof of Stake (PoS)

```go
// Proof of Stake Implementation
type ProofOfStake struct {
    validators map[string]*Validator
    stake      map[string]decimal.Decimal
    mu         sync.RWMutex
}

type Validator struct {
    Address string
    Stake   decimal.Decimal
    Active  bool
}

func NewProofOfStake() *ProofOfStake {
    return &ProofOfStake{
        validators: make(map[string]*Validator),
        stake:      make(map[string]decimal.Decimal),
    }
}

func (pos *ProofOfStake) AddValidator(address string, stake decimal.Decimal) {
    pos.mu.Lock()
    defer pos.mu.Unlock()
    
    validator := &Validator{
        Address: address,
        Stake:   stake,
        Active:  true,
    }
    
    pos.validators[address] = validator
    pos.stake[address] = stake
}

func (pos *ProofOfStake) SelectValidator() string {
    pos.mu.RLock()
    defer pos.mu.RUnlock()
    
    totalStake := decimal.Zero
    for _, stake := range pos.stake {
        totalStake = totalStake.Add(stake)
    }
    
    if totalStake.IsZero() {
        return ""
    }
    
    // Weighted random selection based on stake
    random := rand.Float64()
    cumulative := 0.0
    
    for address, stake := range pos.stake {
        probability := stake.Div(totalStake).InexactFloat64()
        cumulative += probability
        
        if random <= cumulative {
            return address
        }
    }
    
    return ""
}

func (pos *ProofOfStake) ValidateBlock(block *Block, validator string) bool {
    pos.mu.RLock()
    validatorStake := pos.stake[validator]
    pos.mu.RUnlock()
    
    // Check if validator has sufficient stake
    minStake := decimal.NewFromFloat(1000.0)
    return validatorStake.GreaterThanOrEqual(minStake)
}
```

## DeFi (Decentralized Finance)

### Automated Market Maker (AMM)

```go
// Automated Market Maker Implementation
type AMM struct {
    tokenA    string
    tokenB    string
    reserveA  decimal.Decimal
    reserveB  decimal.Decimal
    feeRate   decimal.Decimal
    mu        sync.RWMutex
}

func NewAMM(tokenA, tokenB string, initialA, initialB decimal.Decimal) *AMM {
    return &AMM{
        tokenA:   tokenA,
        tokenB:   tokenB,
        reserveA: initialA,
        reserveB: initialB,
        feeRate:  decimal.NewFromFloat(0.003), // 0.3% fee
    }
}

func (amm *AMM) AddLiquidity(amountA, amountB decimal.Decimal) (decimal.Decimal, error) {
    amm.mu.Lock()
    defer amm.mu.Unlock()
    
    if amm.reserveA.IsZero() && amm.reserveB.IsZero() {
        // Initial liquidity
        amm.reserveA = amountA
        amm.reserveB = amountB
        return amountA, nil
    }
    
    // Calculate required amountB based on current ratio
    requiredB := amountA.Mul(amm.reserveB).Div(amm.reserveA)
    
    if amountB.LessThan(requiredB) {
        return decimal.Zero, fmt.Errorf("insufficient tokenB amount")
    }
    
    // Add liquidity
    amm.reserveA = amm.reserveA.Add(amountA)
    amm.reserveB = amm.reserveB.Add(amountB)
    
    // Calculate LP tokens
    totalSupply := amm.reserveA.Add(amm.reserveB)
    lpTokens := amountA.Add(amountB)
    
    return lpTokens, nil
}

func (amm *AMM) RemoveLiquidity(lpTokens decimal.Decimal) (decimal.Decimal, decimal.Decimal, error) {
    amm.mu.Lock()
    defer amm.mu.Unlock()
    
    totalSupply := amm.reserveA.Add(amm.reserveB)
    
    if lpTokens.GreaterThan(totalSupply) {
        return decimal.Zero, decimal.Zero, fmt.Errorf("insufficient LP tokens")
    }
    
    // Calculate amounts to return
    amountA := lpTokens.Mul(amm.reserveA).Div(totalSupply)
    amountB := lpTokens.Mul(amm.reserveB).Div(totalSupply)
    
    // Update reserves
    amm.reserveA = amm.reserveA.Sub(amountA)
    amm.reserveB = amm.reserveB.Sub(amountB)
    
    return amountA, amountB, nil
}

func (amm *AMM) Swap(tokenIn string, amountIn decimal.Decimal) (decimal.Decimal, error) {
    amm.mu.Lock()
    defer amm.mu.Unlock()
    
    var reserveIn, reserveOut decimal.Decimal
    
    if tokenIn == amm.tokenA {
        reserveIn = amm.reserveA
        reserveOut = amm.reserveB
    } else {
        reserveIn = amm.reserveB
        reserveOut = amm.reserveA
    }
    
    // Calculate amount out using constant product formula
    // amountOut = (amountIn * reserveOut) / (reserveIn + amountIn)
    amountOut := amountIn.Mul(reserveOut).Div(reserveIn.Add(amountIn))
    
    // Apply fee
    fee := amountIn.Mul(amm.feeRate)
    amountInAfterFee := amountIn.Sub(fee)
    
    // Recalculate with fee
    amountOut = amountInAfterFee.Mul(reserveOut).Div(reserveIn.Add(amountInAfterFee))
    
    // Update reserves
    if tokenIn == amm.tokenA {
        amm.reserveA = amm.reserveA.Add(amountIn)
        amm.reserveB = amm.reserveB.Sub(amountOut)
    } else {
        amm.reserveB = amm.reserveB.Add(amountIn)
        amm.reserveA = amm.reserveA.Sub(amountOut)
    }
    
    return amountOut, nil
}
```

### Lending Protocol

```go
// Decentralized Lending Protocol
type LendingProtocol struct {
    markets    map[string]*Market
    users      map[string]*User
    oracle     *PriceOracle
    mu         sync.RWMutex
}

type Market struct {
    Token       string
    TotalSupply decimal.Decimal
    TotalBorrow decimal.Decimal
    Reserve     decimal.Decimal
    InterestRate decimal.Decimal
    CollateralFactor decimal.Decimal
}

type User struct {
    Address     string
    Supplies    map[string]decimal.Decimal
    Borrows     map[string]decimal.Decimal
    Collateral  map[string]decimal.Decimal
}

func NewLendingProtocol() *LendingProtocol {
    return &LendingProtocol{
        markets: make(map[string]*Market),
        users:   make(map[string]*User),
        oracle:  NewPriceOracle(),
    }
}

func (lp *LendingProtocol) Supply(token string, amount decimal.Decimal, user string) error {
    lp.mu.Lock()
    defer lp.mu.Unlock()
    
    market := lp.markets[token]
    if market == nil {
        return fmt.Errorf("market not found")
    }
    
    userAccount := lp.users[user]
    if userAccount == nil {
        userAccount = &User{
            Address:    user,
            Supplies:   make(map[string]decimal.Decimal),
            Borrows:    make(map[string]decimal.Decimal),
            Collateral: make(map[string]decimal.Decimal),
        }
        lp.users[user] = userAccount
    }
    
    // Update market
    market.TotalSupply = market.TotalSupply.Add(amount)
    market.Reserve = market.Reserve.Add(amount.Mul(decimal.NewFromFloat(0.1))) // 10% reserve
    
    // Update user
    userAccount.Supplies[token] = userAccount.Supplies[token].Add(amount)
    userAccount.Collateral[token] = userAccount.Collateral[token].Add(amount)
    
    return nil
}

func (lp *LendingProtocol) Borrow(token string, amount decimal.Decimal, user string) error {
    lp.mu.Lock()
    defer lp.mu.Unlock()
    
    market := lp.markets[token]
    if market == nil {
        return fmt.Errorf("market not found")
    }
    
    userAccount := lp.users[user]
    if userAccount == nil {
        return fmt.Errorf("user not found")
    }
    
    // Check collateral requirements
    if !lp.checkCollateral(userAccount, token, amount) {
        return fmt.Errorf("insufficient collateral")
    }
    
    // Check market liquidity
    availableLiquidity := market.TotalSupply.Sub(market.TotalBorrow).Sub(market.Reserve)
    if amount.GreaterThan(availableLiquidity) {
        return fmt.Errorf("insufficient liquidity")
    }
    
    // Update market
    market.TotalBorrow = market.TotalBorrow.Add(amount)
    
    // Update user
    userAccount.Borrows[token] = userAccount.Borrows[token].Add(amount)
    
    return nil
}

func (lp *LendingProtocol) checkCollateral(user *User, token string, amount decimal.Decimal) bool {
    totalCollateralValue := decimal.Zero
    totalBorrowValue := decimal.Zero
    
    // Calculate total collateral value
    for collateralToken, collateralAmount := range user.Collateral {
        price := lp.oracle.GetPrice(collateralToken)
        market := lp.markets[collateralToken]
        if market != nil {
            collateralValue := collateralAmount.Mul(price).Mul(market.CollateralFactor)
            totalCollateralValue = totalCollateralValue.Add(collateralValue)
        }
    }
    
    // Calculate total borrow value
    for borrowToken, borrowAmount := range user.Borrows {
        price := lp.oracle.GetPrice(borrowToken)
        borrowValue := borrowAmount.Mul(price)
        totalBorrowValue = totalBorrowValue.Add(borrowValue)
    }
    
    // Add new borrow amount
    newBorrowPrice := lp.oracle.GetPrice(token)
    newBorrowValue := amount.Mul(newBorrowPrice)
    totalBorrowValue = totalBorrowValue.Add(newBorrowValue)
    
    // Check if collateral covers borrows (with safety margin)
    safetyMargin := decimal.NewFromFloat(1.5) // 150% collateralization ratio
    requiredCollateral := totalBorrowValue.Mul(safetyMargin)
    
    return totalCollateralValue.GreaterThanOrEqual(requiredCollateral)
}
```

## NFTs and Digital Assets

### NFT Contract

```go
// NFT (Non-Fungible Token) Contract
type NFTContract struct {
    name         string
    symbol       string
    totalSupply  int64
    tokens       map[int64]*NFT
    owners       map[int64]string
    balances     map[string]int64
    approvals    map[int64]string
    mu           sync.RWMutex
}

type NFT struct {
    TokenID   int64
    Owner     string
    Metadata  map[string]interface{}
    URI       string
}

func NewNFTContract(name, symbol string) *NFTContract {
    return &NFTContract{
        name:        name,
        symbol:      symbol,
        totalSupply: 0,
        tokens:      make(map[int64]*NFT),
        owners:      make(map[int64]string),
        balances:    make(map[string]int64),
        approvals:   make(map[int64]string),
    }
}

func (nft *NFTContract) Mint(to string, metadata map[string]interface{}, uri string) (int64, error) {
    nft.mu.Lock()
    defer nft.mu.Unlock()
    
    tokenID := nft.totalSupply + 1
    
    token := &NFT{
        TokenID:  tokenID,
        Owner:    to,
        Metadata: metadata,
        URI:      uri,
    }
    
    nft.tokens[tokenID] = token
    nft.owners[tokenID] = to
    nft.balances[to]++
    nft.totalSupply++
    
    return tokenID, nil
}

func (nft *NFTContract) Transfer(from, to string, tokenID int64) error {
    nft.mu.Lock()
    defer nft.mu.Unlock()
    
    if nft.owners[tokenID] != from {
        return fmt.Errorf("not the owner")
    }
    
    // Update ownership
    nft.owners[tokenID] = to
    nft.tokens[tokenID].Owner = to
    
    // Update balances
    nft.balances[from]--
    nft.balances[to]++
    
    // Clear approval
    delete(nft.approvals, tokenID)
    
    return nil
}

func (nft *NFTContract) Approve(owner, approved string, tokenID int64) error {
    nft.mu.Lock()
    defer nft.mu.Unlock()
    
    if nft.owners[tokenID] != owner {
        return fmt.Errorf("not the owner")
    }
    
    nft.approvals[tokenID] = approved
    return nil
}

func (nft *NFTContract) GetOwner(tokenID int64) string {
    nft.mu.RLock()
    defer nft.mu.RUnlock()
    
    return nft.owners[tokenID]
}

func (nft *NFTContract) GetBalance(owner string) int64 {
    nft.mu.RLock()
    defer nft.mu.RUnlock()
    
    return nft.balances[owner]
}
```

## Web3 Infrastructure

### IPFS Integration

```go
// IPFS (InterPlanetary File System) Integration
type IPFSClient struct {
    gateway string
    client  *http.Client
}

func NewIPFSClient(gateway string) *IPFSClient {
    return &IPFSClient{
        gateway: gateway,
        client:  &http.Client{Timeout: 30 * time.Second},
    }
}

func (ipfs *IPFSClient) Add(data []byte) (string, error) {
    // Upload data to IPFS
    req, err := http.NewRequest("POST", ipfs.gateway+"/api/v0/add", bytes.NewReader(data))
    if err != nil {
        return "", err
    }
    
    req.Header.Set("Content-Type", "application/octet-stream")
    
    resp, err := ipfs.client.Do(req)
    if err != nil {
        return "", err
    }
    defer resp.Body.Close()
    
    var result struct {
        Hash string `json:"Hash"`
    }
    
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return "", err
    }
    
    return result.Hash, nil
}

func (ipfs *IPFSClient) Get(hash string) ([]byte, error) {
    // Retrieve data from IPFS
    resp, err := ipfs.client.Get(ipfs.gateway + "/ipfs/" + hash)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    return ioutil.ReadAll(resp.Body)
}

func (ipfs *IPFSClient) Pin(hash string) error {
    // Pin content to prevent garbage collection
    req, err := http.NewRequest("POST", ipfs.gateway+"/api/v0/pin/add", nil)
    if err != nil {
        return err
    }
    
    q := req.URL.Query()
    q.Add("arg", hash)
    req.URL.RawQuery = q.Encode()
    
    resp, err := ipfs.client.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    return nil
}
```

## Security and Privacy

### Zero-Knowledge Proofs

```go
// Zero-Knowledge Proof System (Simplified)
type ZKProof struct {
    commitment string
    proof      string
    publicInputs []string
}

type ZKVerifier struct {
    publicKey string
}

func NewZKVerifier(publicKey string) *ZKVerifier {
    return &ZKVerifier{
        publicKey: publicKey,
    }
}

func (zkv *ZKVerifier) VerifyProof(proof *ZKProof) bool {
    // Simplified verification - in reality, this would be a full ZK proof verification
    // This is a placeholder for actual zero-knowledge proof verification
    
    // Verify commitment
    if !zkv.verifyCommitment(proof.commitment) {
        return false
    }
    
    // Verify proof
    if !zkv.verifyProof(proof.proof) {
        return false
    }
    
    return true
}

func (zkv *ZKVerifier) verifyCommitment(commitment string) bool {
    // Simplified commitment verification
    return len(commitment) == 64 // Assuming 32-byte commitment
}

func (zkv *ZKVerifier) verifyProof(proof string) bool {
    // Simplified proof verification
    return len(proof) > 0
}

// Privacy-Preserving Transaction
type PrivateTransaction struct {
    Commitment string
    Proof      *ZKProof
    Amount     decimal.Decimal
    Timestamp  time.Time
}

func CreatePrivateTransaction(amount decimal.Decimal, proof *ZKProof) *PrivateTransaction {
    return &PrivateTransaction{
        Commitment: generateCommitment(amount),
        Proof:      proof,
        Amount:     amount,
        Timestamp:  time.Now(),
    }
}

func generateCommitment(amount decimal.Decimal) string {
    // Generate commitment for amount
    data := []byte(amount.String() + time.Now().String())
    hash := sha256.Sum256(data)
    return hex.EncodeToString(hash[:])
}
```

## Conclusion

Blockchain and decentralized systems represent a fundamental shift in how we build and interact with digital systems. Key areas to focus on include:

1. **Blockchain Fundamentals**: Data structures, consensus mechanisms, and cryptographic primitives
2. **Cryptocurrency Systems**: Wallets, transactions, and digital assets
3. **Smart Contracts**: Virtual machines, execution environments, and programming models
4. **Decentralized Applications**: Frontend integration, Web3 protocols, and user experience
5. **Consensus Mechanisms**: Proof of Work, Proof of Stake, and other consensus algorithms
6. **DeFi**: Automated market makers, lending protocols, and financial primitives
7. **NFTs**: Digital asset standards, metadata, and ownership models
8. **Web3 Infrastructure**: IPFS, decentralized storage, and peer-to-peer networks
9. **Security and Privacy**: Zero-knowledge proofs, privacy-preserving transactions, and cryptographic security

Mastering these areas will prepare you for building next-generation decentralized applications and understanding the future of digital systems.

## Additional Resources

- [Ethereum Documentation](https://ethereum.org/en/developers/docs/)
- [Bitcoin Whitepaper](https://bitcoin.org/bitcoin.pdf)
- [Solidity Documentation](https://docs.soliditylang.org/)
- [Web3.js Documentation](https://web3js.readthedocs.io/)
- [IPFS Documentation](https://docs.ipfs.io/)
- [DeFi Pulse](https://defipulse.com/)
- [OpenZeppelin Contracts](https://openzeppelin.com/contracts/)
- [ConsenSys Academy](https://consensys.net/academy/)
