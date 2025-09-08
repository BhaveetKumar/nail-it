# 🌐 Distributed Systems: Building Scalable and Reliable Systems

> **Complete guide to distributed systems design, patterns, and best practices**

## 🎯 **Learning Objectives**

- Master distributed systems fundamentals and challenges
- Understand consensus algorithms and distributed coordination
- Implement fault tolerance and reliability patterns
- Design scalable distributed architectures
- Handle distributed data consistency and partitioning

## 📚 **Table of Contents**

1. [Distributed Systems Fundamentals](#distributed-systems-fundamentals)
2. [Consensus Algorithms](#consensus-algorithms)
3. [Distributed Coordination](#distributed-coordination)
4. [Fault Tolerance](#fault-tolerance)
5. [Data Consistency](#data-consistency)
6. [Scalability Patterns](#scalability-patterns)
7. [Interview Questions](#interview-questions)

---

## 🌐 **Distributed Systems Fundamentals**

### **Concept**

Distributed systems are collections of independent computers that appear to users as a single coherent system. They provide scalability, fault tolerance, and performance benefits but introduce complexity in coordination, consistency, and failure handling.

### **Key Characteristics**

1. **Concurrency**: Multiple processes running simultaneously
2. **No Global Clock**: Independent timing across nodes
3. **Independent Failures**: Nodes can fail independently
4. **Heterogeneity**: Different hardware and software
5. **Transparency**: Hide complexity from users
6. **Scalability**: Handle increasing load and data

### **Challenges**

- **Network Partitions**: Network failures can split the system
- **Consistency**: Maintaining data consistency across nodes
- **Coordination**: Synchronizing actions across distributed nodes
- **Failure Handling**: Detecting and recovering from failures
- **Security**: Securing distributed communications
- **Performance**: Optimizing across network boundaries

---

## 🤝 **Consensus Algorithms**

### **1. Raft Algorithm**

**Concept**: A consensus algorithm for managing replicated logs in distributed systems.

**Code Example**:
```python
import asyncio
import time
import random
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
import json

class NodeState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"

@dataclass
class LogEntry:
    term: int
    index: int
    command: Any

@dataclass
class RaftRequest:
    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: List[LogEntry]
    leader_commit: int

@dataclass
class RaftResponse:
    term: int
    success: bool

class RaftNode:
    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = peers
        self.state = NodeState.FOLLOWER
        
        # Persistent state
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        
        # Volatile state
        self.commit_index = 0
        self.last_applied = 0
        
        # Leader state
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        # Election timeout
        self.election_timeout = random.uniform(1.0, 2.0)
        self.last_heartbeat = time.time()
        
        # Heartbeat interval
        self.heartbeat_interval = 0.1
        
    async def start(self):
        """Start the Raft node"""
        print(f"Node {self.node_id} starting...")
        
        # Start election timer
        asyncio.create_task(self._election_timer())
        
        # Start heartbeat if leader
        asyncio.create_task(self._heartbeat_timer())
        
        # Start message processing
        asyncio.create_task(self._process_messages())
    
    async def _election_timer(self):
        """Election timeout timer"""
        while True:
            await asyncio.sleep(self.election_timeout)
            
            if self.state == NodeState.FOLLOWER:
                # Start election
                await self._start_election()
            elif self.state == NodeState.CANDIDATE:
                # Election timeout, start new election
                await self._start_election()
    
    async def _start_election(self):
        """Start leader election"""
        print(f"Node {self.node_id} starting election for term {self.current_term + 1}")
        
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.election_timeout = random.uniform(1.0, 2.0)
        
        # Request votes from peers
        votes_received = 1  # Vote for self
        total_votes = len(self.peers) + 1
        
        for peer in self.peers:
            try:
                response = await self._request_vote(peer)
                if response and response.term > self.current_term:
                    self.current_term = response.term
                    self.state = NodeState.FOLLOWER
                    return
                elif response and response.success:
                    votes_received += 1
            except Exception as e:
                print(f"Error requesting vote from {peer}: {e}")
        
        # Check if we won the election
        if votes_received > total_votes // 2:
            await self._become_leader()
        else:
            self.state = NodeState.FOLLOWER
    
    async def _request_vote(self, peer: str) -> Optional[RaftResponse]:
        """Request vote from peer"""
        # Mock implementation - in real system, this would be RPC
        await asyncio.sleep(0.01)  # Simulate network delay
        
        # Mock response
        return RaftResponse(term=self.current_term, success=True)
    
    async def _become_leader(self):
        """Become leader"""
        print(f"Node {self.node_id} became leader for term {self.current_term}")
        
        self.state = NodeState.LEADER
        
        # Initialize leader state
        for peer in self.peers:
            self.next_index[peer] = len(self.log)
            self.match_index[peer] = 0
    
    async def _heartbeat_timer(self):
        """Send heartbeats if leader"""
        while True:
            if self.state == NodeState.LEADER:
                await self._send_heartbeats()
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _send_heartbeats(self):
        """Send heartbeats to followers"""
        for peer in self.peers:
            try:
                await self._append_entries(peer, [])
            except Exception as e:
                print(f"Error sending heartbeat to {peer}: {e}")
    
    async def _append_entries(self, peer: str, entries: List[LogEntry]):
        """Send append entries to peer"""
        # Mock implementation - in real system, this would be RPC
        await asyncio.sleep(0.01)  # Simulate network delay
        
        # Mock response
        return RaftResponse(term=self.current_term, success=True)
    
    async def _process_messages(self):
        """Process incoming messages"""
        while True:
            # Mock message processing
            await asyncio.sleep(0.1)
    
    async def append_log(self, command: Any):
        """Append log entry (leader only)"""
        if self.state != NodeState.LEADER:
            raise Exception("Only leader can append log entries")
        
        # Create log entry
        entry = LogEntry(
            term=self.current_term,
            index=len(self.log),
            command=command
        )
        
        self.log.append(entry)
        
        # Replicate to followers
        for peer in self.peers:
            try:
                await self._append_entries(peer, [entry])
            except Exception as e:
                print(f"Error replicating to {peer}: {e}")
    
    def get_log(self) -> List[LogEntry]:
        """Get committed log entries"""
        return self.log[:self.commit_index + 1]

# Example usage
async def raft_example():
    """Example of Raft consensus"""
    # Create nodes
    nodes = []
    node_ids = ["node1", "node2", "node3"]
    
    for node_id in node_ids:
        peers = [nid for nid in node_ids if nid != node_id]
        node = RaftNode(node_id, peers)
        nodes.append(node)
    
    # Start all nodes
    tasks = []
    for node in nodes:
        task = asyncio.create_task(node.start())
        tasks.append(task)
    
    # Wait for leader election
    await asyncio.sleep(3)
    
    # Find leader and append log
    leader = None
    for node in nodes:
        if node.state == NodeState.LEADER:
            leader = node
            break
    
    if leader:
        print(f"Leader found: {leader.node_id}")
        await leader.append_log("test_command")
        print(f"Log appended. Current log: {leader.get_log()}")
    
    # Cleanup
    for task in tasks:
        task.cancel()

if __name__ == "__main__":
    asyncio.run(raft_example())
```

### **2. Paxos Algorithm**

**Code Example**:
```python
class PaxosNode:
    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = peers
        
        # Paxos state
        self.proposal_number = 0
        self.accepted_proposal = None
        self.accepted_value = None
        self.learned_value = None
        
        # Ballot number
        self.ballot_number = 0
    
    async def propose(self, value: Any) -> Optional[Any]:
        """Propose a value using Paxos"""
        while True:
            # Phase 1: Prepare
            ballot = self._generate_ballot()
            prepare_responses = await self._prepare_phase(ballot)
            
            if len(prepare_responses) > len(self.peers) // 2:
                # Phase 2: Accept
                accept_responses = await self._accept_phase(ballot, value)
                
                if len(accept_responses) > len(self.peers) // 2:
                    # Value accepted
                    self.learned_value = value
                    return value
            
            # Retry with higher ballot number
            await asyncio.sleep(0.1)
    
    def _generate_ballot(self) -> int:
        """Generate unique ballot number"""
        self.ballot_number += 1
        return self.ballot_number
    
    async def _prepare_phase(self, ballot: int) -> List[Dict[str, Any]]:
        """Phase 1: Prepare"""
        responses = []
        
        for peer in self.peers:
            try:
                response = await self._send_prepare(peer, ballot)
                if response:
                    responses.append(response)
            except Exception as e:
                print(f"Error in prepare phase with {peer}: {e}")
        
        return responses
    
    async def _accept_phase(self, ballot: int, value: Any) -> List[Dict[str, Any]]:
        """Phase 2: Accept"""
        responses = []
        
        for peer in self.peers:
            try:
                response = await self._send_accept(peer, ballot, value)
                if response:
                    responses.append(response)
            except Exception as e:
                print(f"Error in accept phase with {peer}: {e}")
        
        return responses
    
    async def _send_prepare(self, peer: str, ballot: int) -> Optional[Dict[str, Any]]:
        """Send prepare message"""
        # Mock implementation
        await asyncio.sleep(0.01)
        return {"ballot": ballot, "accepted_proposal": self.accepted_proposal, "accepted_value": self.accepted_value}
    
    async def _send_accept(self, peer: str, ballot: int, value: Any) -> Optional[Dict[str, Any]]:
        """Send accept message"""
        # Mock implementation
        await asyncio.sleep(0.01)
        return {"ballot": ballot, "accepted": True}

# Example usage
async def paxos_example():
    """Example of Paxos consensus"""
    # Create nodes
    nodes = []
    node_ids = ["node1", "node2", "node3"]
    
    for node_id in node_ids:
        peers = [nid for nid in node_ids if nid != node_id]
        node = PaxosNode(node_id, peers)
        nodes.append(node)
    
    # Propose values
    tasks = []
    for i, node in enumerate(nodes):
        task = asyncio.create_task(node.propose(f"value_{i}"))
        tasks.append(task)
    
    # Wait for consensus
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    print("Paxos Results:")
    for i, result in enumerate(results):
        print(f"Node {i}: {result}")

if __name__ == "__main__":
    asyncio.run(paxos_example())
```

---

## 🔄 **Distributed Coordination**

### **1. Distributed Locks**

**Code Example**:
```python
import asyncio
import time
import uuid
from typing import Dict, Any, Optional
import redis

class DistributedLock:
    def __init__(self, redis_client: redis.Redis, lock_name: str, timeout: int = 10):
        self.redis_client = redis_client
        self.lock_name = lock_name
        self.timeout = timeout
        self.identifier = str(uuid.uuid4())
        self.acquired = False
    
    async def acquire(self) -> bool:
        """Acquire distributed lock"""
        end_time = time.time() + self.timeout
        
        while time.time() < end_time:
            # Try to acquire lock
            if self.redis_client.set(self.lock_name, self.identifier, nx=True, ex=self.timeout):
                self.acquired = True
                return True
            
            # Wait before retry
            await asyncio.sleep(0.1)
        
        return False
    
    async def release(self) -> bool:
        """Release distributed lock"""
        if not self.acquired:
            return False
        
        # Use Lua script to ensure atomic release
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        
        result = self.redis_client.eval(lua_script, 1, self.lock_name, self.identifier)
        self.acquired = False
        return result == 1
    
    async def extend(self, additional_time: int) -> bool:
        """Extend lock timeout"""
        if not self.acquired:
            return False
        
        # Use Lua script to ensure atomic extension
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("expire", KEYS[1], ARGV[2])
        else
            return 0
        end
        """
        
        result = self.redis_client.eval(lua_script, 1, self.lock_name, self.identifier, additional_time)
        return result == 1

# Example usage
async def distributed_lock_example():
    """Example of distributed lock usage"""
    # Mock Redis client
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    # Create lock
    lock = DistributedLock(redis_client, "my_lock", timeout=10)
    
    # Acquire lock
    if await lock.acquire():
        print("Lock acquired successfully")
        
        # Do some work
        await asyncio.sleep(2)
        
        # Release lock
        if await lock.release():
            print("Lock released successfully")
    else:
        print("Failed to acquire lock")

if __name__ == "__main__":
    asyncio.run(distributed_lock_example())
```

### **2. Leader Election**

**Code Example**:
```python
class LeaderElection:
    def __init__(self, redis_client: redis.Redis, election_key: str, node_id: str, ttl: int = 30):
        self.redis_client = redis_client
        self.election_key = election_key
        self.node_id = node_id
        self.ttl = ttl
        self.is_leader = False
        self.heartbeat_interval = 5
    
    async def start_election(self):
        """Start leader election process"""
        while True:
            try:
                # Try to become leader
                if await self._try_become_leader():
                    self.is_leader = True
                    print(f"Node {self.node_id} became leader")
                    
                    # Start heartbeat
                    await self._heartbeat_loop()
                else:
                    # Check if current leader is alive
                    current_leader = await self._get_current_leader()
                    if not current_leader:
                        print(f"Node {self.node_id} detected no leader, retrying...")
                    else:
                        print(f"Node {self.node_id} following leader {current_leader}")
                
                # Wait before next election attempt
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error in leader election: {e}")
                await asyncio.sleep(1)
    
    async def _try_become_leader(self) -> bool:
        """Try to become leader"""
        # Use SET with NX and EX for atomic leader election
        result = self.redis_client.set(
            self.election_key, 
            self.node_id, 
            nx=True,  # Only set if not exists
            ex=self.ttl  # Expire after TTL
        )
        
        return result is not None
    
    async def _get_current_leader(self) -> Optional[str]:
        """Get current leader"""
        leader = self.redis_client.get(self.election_key)
        return leader.decode('utf-8') if leader else None
    
    async def _heartbeat_loop(self):
        """Send heartbeats to maintain leadership"""
        while self.is_leader:
            try:
                # Extend leadership
                result = self.redis_client.expire(self.election_key, self.ttl)
                
                if not result:
                    # Lost leadership
                    self.is_leader = False
                    print(f"Node {self.node_id} lost leadership")
                    break
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                print(f"Error in heartbeat: {e}")
                self.is_leader = False
                break
    
    async def stop(self):
        """Stop leader election"""
        if self.is_leader:
            # Release leadership
            current_leader = await self._get_current_leader()
            if current_leader == self.node_id:
                self.redis_client.delete(self.election_key)
            self.is_leader = False

# Example usage
async def leader_election_example():
    """Example of leader election"""
    # Mock Redis client
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    # Create leader election
    election = LeaderElection(redis_client, "leader_election", "node1", ttl=30)
    
    # Start election
    task = asyncio.create_task(election.start_election())
    
    # Run for some time
    await asyncio.sleep(10)
    
    # Stop election
    await election.stop()
    task.cancel()

if __name__ == "__main__":
    asyncio.run(leader_election_example())
```

---

## 🛡️ **Fault Tolerance**

### **1. Circuit Breaker Pattern**

**Code Example**:
```python
import asyncio
import time
from enum import Enum
from typing import Callable, Any, Optional
from dataclasses import dataclass

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 3

class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.next_attempt_time = None
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if time.time() < self.next_attempt_time:
                raise Exception("Circuit breaker is OPEN")
            else:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e
    
    async def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = 0
    
    async def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.next_attempt_time = time.time() + self.config.recovery_timeout
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "next_attempt_time": self.next_attempt_time
        }

# Example usage
async def circuit_breaker_example():
    """Example of circuit breaker usage"""
    config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=5)
    circuit_breaker = CircuitBreaker(config)
    
    async def unreliable_service():
        """Mock unreliable service"""
        import random
        if random.random() < 0.7:  # 70% failure rate
            raise Exception("Service unavailable")
        return "Success"
    
    # Test circuit breaker
    for i in range(10):
        try:
            result = await circuit_breaker.call(unreliable_service)
            print(f"Call {i}: {result}")
        except Exception as e:
            print(f"Call {i}: {e}")
        
        print(f"Circuit state: {circuit_breaker.get_state()}")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(circuit_breaker_example())
```

### **2. Retry Pattern**

**Code Example**:
```python
import asyncio
import random
from typing import Callable, Any, Optional
from dataclasses import dataclass

@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

class RetryHandler:
    def __init__(self, config: RetryConfig):
        self.config = config
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.config.max_attempts - 1:
                    # Last attempt, re-raise exception
                    raise e
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry"""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            # Add random jitter to prevent thundering herd
            delay *= (0.5 + random.random() * 0.5)
        
        return delay

# Example usage
async def retry_example():
    """Example of retry pattern usage"""
    config = RetryConfig(max_attempts=5, base_delay=1.0, max_delay=10.0)
    retry_handler = RetryHandler(config)
    
    async def unreliable_service():
        """Mock unreliable service"""
        import random
        if random.random() < 0.8:  # 80% failure rate
            raise Exception("Service temporarily unavailable")
        return "Success"
    
    try:
        result = await retry_handler.execute(unreliable_service)
        print(f"Final result: {result}")
    except Exception as e:
        print(f"All retry attempts failed: {e}")

if __name__ == "__main__":
    asyncio.run(retry_example())
```

---

## 🎯 **Interview Questions**

### **1. What are the main challenges in distributed systems?**

**Answer:**
- **Network Partitions**: Network failures can split the system
- **Consistency**: Maintaining data consistency across nodes
- **Coordination**: Synchronizing actions across distributed nodes
- **Failure Handling**: Detecting and recovering from failures
- **Security**: Securing distributed communications
- **Performance**: Optimizing across network boundaries

### **2. Explain the CAP theorem.**

**Answer:**
- **Consistency**: All nodes see the same data simultaneously
- **Availability**: System remains operational
- **Partition Tolerance**: System continues despite network failures
- **Trade-off**: Can only guarantee 2 out of 3 properties
- **Examples**: CP (Consistency + Partition Tolerance), AP (Availability + Partition Tolerance)

### **3. What's the difference between Raft and Paxos?**

**Answer:**
- **Raft**: Easier to understand, leader-based, good for practical systems
- **Paxos**: More complex, leaderless, theoretically optimal
- **Raft**: Better for implementation and debugging
- **Paxos**: Better for theoretical analysis and optimization

### **4. How do you handle split-brain scenarios?**

**Answer:**
- **Quorum-based**: Require majority of nodes for decisions
- **Fencing**: Prevent access to shared resources
- **Witness Nodes**: Use external nodes to break ties
- **Time-based**: Use synchronized clocks for ordering
- **Consensus**: Use consensus algorithms like Raft or Paxos

### **5. What are the benefits and drawbacks of microservices?**

**Answer:**
- **Benefits**: Scalability, technology diversity, fault isolation, team autonomy
- **Drawbacks**: Complexity, network overhead, data consistency, debugging difficulty
- **Trade-offs**: Better for large teams and complex systems, worse for simple applications

---

**🎉 Distributed systems require careful design to handle complexity while providing scalability and reliability!**
