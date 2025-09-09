# 📡 Event-Driven Architecture: Building Reactive Systems

> **Complete guide to event-driven architecture, patterns, and implementation**

## 🎯 **Learning Objectives**

- Master event-driven architecture fundamentals
- Understand event sourcing and CQRS patterns
- Implement event streaming and message queues
- Build reactive and resilient systems
- Handle event processing and orchestration

## 📚 **Table of Contents**

1. [Event-Driven Architecture Fundamentals](#event-driven-architecture-fundamentals)
2. [Event Sourcing](#event-sourcing)
3. [CQRS Pattern](#cqrs-pattern)
4. [Event Streaming](#event-streaming)
5. [Message Queues](#message-queues)
6. [Event Processing](#event-processing)
7. [Interview Questions](#interview-questions)

---

## 📡 **Event-Driven Architecture Fundamentals**

### **Concept**

Event-driven architecture (EDA) is a design pattern where system components communicate through events. Events represent significant business occurrences and trigger reactions in other parts of the system, enabling loose coupling and high scalability.

### **Key Components**

1. **Event Producers**: Generate and publish events
2. **Event Consumers**: Subscribe to and process events
3. **Event Bus**: Routes events between producers and consumers
4. **Event Store**: Persists events for replay and audit
5. **Event Handlers**: Process specific event types
6. **Event Sinks**: Store processed event results

### **Benefits**

- **Loose Coupling**: Components don't need direct knowledge of each other
- **Scalability**: Easy to add new consumers and scale independently
- **Resilience**: System continues working even if some components fail
- **Flexibility**: Easy to add new features and modify existing ones
- **Auditability**: Complete event history for debugging and compliance

---

## 📝 **Event Sourcing**

### **1. Event Store Implementation**

**Code Example**:
```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import uuid
from enum import Enum

class EventType(Enum):
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    ORDER_PLACED = "order_placed"
    ORDER_CANCELLED = "order_cancelled"
    PAYMENT_PROCESSED = "payment_processed"

@dataclass
class Event:
    event_id: str
    event_type: EventType
    aggregate_id: str
    aggregate_type: str
    event_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1

class EventStore:
    def __init__(self, storage_path: str = "./event_store"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.events: List[Event] = []
        self.aggregate_versions: Dict[str, int] = {}
    
    def append_event(self, event: Event) -> bool:
        """Append event to store"""
        # Check version for optimistic concurrency control
        if event.aggregate_id in self.aggregate_versions:
            expected_version = self.aggregate_versions[event.aggregate_id]
            if event.version != expected_version + 1:
                raise ValueError(f"Version mismatch for aggregate {event.aggregate_id}")
        
        # Add event
        self.events.append(event)
        self.aggregate_versions[event.aggregate_id] = event.version
        
        # Persist to disk
        self._persist_event(event)
        
        return True
    
    def get_events(self, aggregate_id: str, from_version: int = 0) -> List[Event]:
        """Get events for specific aggregate"""
        return [
            event for event in self.events
            if event.aggregate_id == aggregate_id and event.version > from_version
        ]
    
    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """Get events by type"""
        return [event for event in self.events if event.event_type == event_type]
    
    def get_events_since(self, timestamp: datetime) -> List[Event]:
        """Get events since timestamp"""
        return [event for event in self.events if event.timestamp >= timestamp]
    
    def _persist_event(self, event: Event):
        """Persist event to disk"""
        event_file = self.storage_path / f"{event.aggregate_id}_{event.version}.json"
        
        event_data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "aggregate_id": event.aggregate_id,
            "aggregate_type": event.aggregate_type,
            "event_data": event.event_data,
            "metadata": event.metadata,
            "timestamp": event.timestamp.isoformat(),
            "version": event.version
        }
        
        with open(event_file, 'w') as f:
            json.dump(event_data, f, indent=2)
    
    def replay_events(self, aggregate_id: str) -> List[Event]:
        """Replay all events for aggregate"""
        return self.get_events(aggregate_id, 0)

# Example usage
def event_sourcing_example():
    """Example of event sourcing"""
    event_store = EventStore()
    
    # Create events
    user_created = Event(
        event_id=str(uuid.uuid4()),
        event_type=EventType.USER_CREATED,
        aggregate_id="user_123",
        aggregate_type="User",
        event_data={"name": "John Doe", "email": "john@example.com"},
        version=1
    )
    
    user_updated = Event(
        event_id=str(uuid.uuid4()),
        event_type=EventType.USER_UPDATED,
        aggregate_id="user_123",
        aggregate_type="User",
        event_data={"name": "John Smith", "email": "john.smith@example.com"},
        version=2
    )
    
    # Append events
    event_store.append_event(user_created)
    event_store.append_event(user_updated)
    
    # Get events
    events = event_store.get_events("user_123")
    print(f"Found {len(events)} events for user_123")
    
    for event in events:
        print(f"  {event.event_type.value}: {event.event_data}")

if __name__ == "__main__":
    event_sourcing_example()
```

### **2. Aggregate Root Implementation**

**Code Example**:
```python
class UserAggregate:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.name = ""
        self.email = ""
        self.is_active = True
        self.version = 0
        self.uncommitted_events: List[Event] = []
    
    def create_user(self, name: str, email: str):
        """Create new user"""
        if self.version > 0:
            raise ValueError("User already exists")
        
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=EventType.USER_CREATED,
            aggregate_id=self.user_id,
            aggregate_type="User",
            event_data={"name": name, "email": email},
            version=self.version + 1
        )
        
        self.apply_event(event)
        self.uncommitted_events.append(event)
    
    def update_user(self, name: str = None, email: str = None):
        """Update user"""
        if self.version == 0:
            raise ValueError("User does not exist")
        
        event_data = {}
        if name is not None:
            event_data["name"] = name
        if email is not None:
            event_data["email"] = email
        
        if not event_data:
            return
        
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=EventType.USER_UPDATED,
            aggregate_id=self.user_id,
            aggregate_type="User",
            event_data=event_data,
            version=self.version + 1
        )
        
        self.apply_event(event)
        self.uncommitted_events.append(event)
    
    def delete_user(self):
        """Delete user"""
        if self.version == 0:
            raise ValueError("User does not exist")
        
        if not self.is_active:
            return
        
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=EventType.USER_DELETED,
            aggregate_id=self.user_id,
            aggregate_type="User",
            event_data={},
            version=self.version + 1
        )
        
        self.apply_event(event)
        self.uncommitted_events.append(event)
    
    def apply_event(self, event: Event):
        """Apply event to aggregate"""
        if event.event_type == EventType.USER_CREATED:
            self.name = event.event_data["name"]
            self.email = event.event_data["email"]
            self.is_active = True
        elif event.event_type == EventType.USER_UPDATED:
            if "name" in event.event_data:
                self.name = event.event_data["name"]
            if "email" in event.event_data:
                self.email = event.event_data["email"]
        elif event.event_type == EventType.USER_DELETED:
            self.is_active = False
        
        self.version = event.version
    
    def get_uncommitted_events(self) -> List[Event]:
        """Get uncommitted events"""
        return self.uncommitted_events.copy()
    
    def mark_events_as_committed(self):
        """Mark events as committed"""
        self.uncommitted_events.clear()
    
    @classmethod
    def from_events(cls, user_id: str, events: List[Event]) -> 'UserAggregate':
        """Reconstruct aggregate from events"""
        aggregate = cls(user_id)
        
        for event in sorted(events, key=lambda e: e.version):
            aggregate.apply_event(event)
        
        return aggregate

# Example usage
def aggregate_example():
    """Example of aggregate usage"""
    # Create user
    user = UserAggregate("user_123")
    user.create_user("John Doe", "john@example.com")
    
    print(f"User created: {user.name} ({user.email})")
    
    # Update user
    user.update_user(name="John Smith", email="john.smith@example.com")
    print(f"User updated: {user.name} ({user.email})")
    
    # Get uncommitted events
    events = user.get_uncommitted_events()
    print(f"Uncommitted events: {len(events)}")
    
    # Mark as committed
    user.mark_events_as_committed()
    print("Events marked as committed")

if __name__ == "__main__":
    aggregate_example()
```

---

## 🔄 **CQRS Pattern**

### **1. Command and Query Separation**

**Code Example**:
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass

@dataclass
class Command:
    command_id: str
    aggregate_id: str
    command_data: Dict[str, Any]

@dataclass
class Query:
    query_id: str
    query_data: Dict[str, Any]

@dataclass
class CommandResult:
    success: bool
    aggregate_id: str
    version: int
    events: List[Event]
    error: Optional[str] = None

@dataclass
class QueryResult:
    success: bool
    data: Any
    error: Optional[str] = None

class CommandHandler(ABC):
    @abstractmethod
    async def handle(self, command: Command) -> CommandResult:
        pass

class QueryHandler(ABC):
    @abstractmethod
    async def handle(self, query: Query) -> QueryResult:
        pass

class CreateUserCommand(Command):
    def __init__(self, user_id: str, name: str, email: str):
        super().__init__(
            command_id=str(uuid.uuid4()),
            aggregate_id=user_id,
            command_data={"name": name, "email": email}
        )

class UpdateUserCommand(Command):
    def __init__(self, user_id: str, name: str = None, email: str = None):
        super().__init__(
            command_id=str(uuid.uuid4()),
            aggregate_id=user_id,
            command_data={"name": name, "email": email}
        )

class GetUserQuery(Query):
    def __init__(self, user_id: str):
        super().__init__(
            query_id=str(uuid.uuid4()),
            query_data={"user_id": user_id}
        )

class UserCommandHandler(CommandHandler):
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
    
    async def handle(self, command: Command) -> CommandResult:
        try:
            if isinstance(command, CreateUserCommand):
                return await self._handle_create_user(command)
            elif isinstance(command, UpdateUserCommand):
                return await self._handle_update_user(command)
            else:
                return CommandResult(
                    success=False,
                    aggregate_id=command.aggregate_id,
                    version=0,
                    events=[],
                    error="Unknown command type"
                )
        except Exception as e:
            return CommandResult(
                success=False,
                aggregate_id=command.aggregate_id,
                version=0,
                events=[],
                error=str(e)
            )
    
    async def _handle_create_user(self, command: CreateUserCommand) -> CommandResult:
        """Handle create user command"""
        # Check if user already exists
        existing_events = self.event_store.get_events(command.aggregate_id)
        if existing_events:
            return CommandResult(
                success=False,
                aggregate_id=command.aggregate_id,
                version=0,
                events=[],
                error="User already exists"
            )
        
        # Create user aggregate
        user = UserAggregate(command.aggregate_id)
        user.create_user(
            command.command_data["name"],
            command.command_data["email"]
        )
        
        # Get uncommitted events
        events = user.get_uncommitted_events()
        
        # Append events to store
        for event in events:
            self.event_store.append_event(event)
        
        user.mark_events_as_committed()
        
        return CommandResult(
            success=True,
            aggregate_id=command.aggregate_id,
            version=user.version,
            events=events
        )
    
    async def _handle_update_user(self, command: UpdateUserCommand) -> CommandResult:
        """Handle update user command"""
        # Get existing events
        existing_events = self.event_store.get_events(command.aggregate_id)
        if not existing_events:
            return CommandResult(
                success=False,
                aggregate_id=command.aggregate_id,
                version=0,
                events=[],
                error="User does not exist"
            )
        
        # Reconstruct aggregate
        user = UserAggregate.from_events(command.aggregate_id, existing_events)
        
        # Update user
        user.update_user(
            command.command_data.get("name"),
            command.command_data.get("email")
        )
        
        # Get uncommitted events
        events = user.get_uncommitted_events()
        
        # Append events to store
        for event in events:
            self.event_store.append_event(event)
        
        user.mark_events_as_committed()
        
        return CommandResult(
            success=True,
            aggregate_id=command.aggregate_id,
            version=user.version,
            events=events
        )

class UserQueryHandler(QueryHandler):
    def __init__(self, read_model: Dict[str, Any]):
        self.read_model = read_model
    
    async def handle(self, query: Query) -> QueryResult:
        try:
            if isinstance(query, GetUserQuery):
                return await self._handle_get_user(query)
            else:
                return QueryResult(
                    success=False,
                    data=None,
                    error="Unknown query type"
                )
        except Exception as e:
            return QueryResult(
                success=False,
                data=None,
                error=str(e)
            )
    
    async def _handle_get_user(self, query: GetUserQuery) -> QueryResult:
        """Handle get user query"""
        user_id = query.query_data["user_id"]
        
        if user_id not in self.read_model:
            return QueryResult(
                success=False,
                data=None,
                error="User not found"
            )
        
        return QueryResult(
            success=True,
            data=self.read_model[user_id]
        )

# Example usage
async def cqrs_example():
    """Example of CQRS pattern"""
    event_store = EventStore()
    read_model = {}
    
    # Create command handler
    command_handler = UserCommandHandler(event_store)
    
    # Create query handler
    query_handler = UserQueryHandler(read_model)
    
    # Create user command
    create_command = CreateUserCommand("user_123", "John Doe", "john@example.com")
    result = await command_handler.handle(create_command)
    
    if result.success:
        print(f"User created successfully: {result.aggregate_id}")
        
        # Update read model
        read_model[result.aggregate_id] = {
            "name": "John Doe",
            "email": "john@example.com",
            "is_active": True
        }
    else:
        print(f"Failed to create user: {result.error}")
    
    # Query user
    get_query = GetUserQuery("user_123")
    query_result = await query_handler.handle(get_query)
    
    if query_result.success:
        print(f"User found: {query_result.data}")
    else:
        print(f"Failed to get user: {query_result.error}")

if __name__ == "__main__":
    asyncio.run(cqrs_example())
```

---

## 🌊 **Event Streaming**

### **1. Event Stream Processing**

**Code Example**:
```python
import asyncio
from typing import Dict, Any, List, Callable
from collections import deque
import time

class EventStream:
    def __init__(self, name: str, max_size: int = 10000):
        self.name = name
        self.max_size = max_size
        self.events: deque = deque(maxlen=max_size)
        self.subscribers: List[Callable] = []
        self.running = False
    
    def publish(self, event: Event):
        """Publish event to stream"""
        self.events.append(event)
        
        # Notify subscribers
        for subscriber in self.subscribers:
            try:
                subscriber(event)
            except Exception as e:
                print(f"Error in subscriber: {e}")
    
    def subscribe(self, handler: Callable):
        """Subscribe to stream"""
        self.subscribers.append(handler)
    
    def get_events_since(self, timestamp: datetime) -> List[Event]:
        """Get events since timestamp"""
        return [event for event in self.events if event.timestamp >= timestamp]
    
    def get_latest_events(self, count: int) -> List[Event]:
        """Get latest events"""
        return list(self.events)[-count:]

class EventStreamProcessor:
    def __init__(self):
        self.streams: Dict[str, EventStream] = {}
        self.processors: List[Callable] = []
    
    def create_stream(self, name: str, max_size: int = 10000) -> EventStream:
        """Create new event stream"""
        stream = EventStream(name, max_size)
        self.streams[name] = stream
        return stream
    
    def get_stream(self, name: str) -> EventStream:
        """Get event stream"""
        return self.streams.get(name)
    
    def add_processor(self, processor: Callable):
        """Add event processor"""
        self.processors.append(processor)
    
    async def start_processing(self):
        """Start event processing"""
        while True:
            for processor in self.processors:
                try:
                    await processor()
                except Exception as e:
                    print(f"Error in processor: {e}")
            
            await asyncio.sleep(0.1)  # Process every 100ms

# Example usage
async def event_streaming_example():
    """Example of event streaming"""
    processor = EventStreamProcessor()
    
    # Create streams
    user_stream = processor.create_stream("user_events")
    order_stream = processor.create_stream("order_events")
    
    # Add processors
    async def user_event_processor():
        """Process user events"""
        events = user_stream.get_latest_events(10)
        for event in events:
            print(f"Processing user event: {event.event_type.value}")
    
    async def order_event_processor():
        """Process order events"""
        events = order_stream.get_latest_events(10)
        for event in events:
            print(f"Processing order event: {event.event_type.value}")
    
    processor.add_processor(user_event_processor)
    processor.add_processor(order_event_processor)
    
    # Start processing
    processing_task = asyncio.create_task(processor.start_processing())
    
    # Publish some events
    user_created = Event(
        event_id=str(uuid.uuid4()),
        event_type=EventType.USER_CREATED,
        aggregate_id="user_123",
        aggregate_type="User",
        event_data={"name": "John Doe", "email": "john@example.com"}
    )
    
    user_stream.publish(user_created)
    
    # Wait a bit
    await asyncio.sleep(1)
    
    # Stop processing
    processing_task.cancel()

if __name__ == "__main__":
    asyncio.run(event_streaming_example())
```

---

## 🎯 **Interview Questions**

### **1. What is event-driven architecture and what are its benefits?**

**Answer:**
- **Definition**: Architecture where components communicate through events
- **Benefits**: Loose coupling, scalability, resilience, flexibility, auditability
- **Use Cases**: Microservices, real-time systems, event sourcing
- **Challenges**: Event ordering, duplicate processing, eventual consistency
- **Patterns**: Event sourcing, CQRS, saga pattern

### **2. Explain the difference between event sourcing and traditional CRUD.**

**Answer:**
- **Event Sourcing**: Store events that represent state changes
- **Traditional CRUD**: Store current state directly
- **Benefits**: Complete audit trail, time travel, replay capability
- **Drawbacks**: Complexity, storage requirements, eventual consistency
- **Use Cases**: Financial systems, audit requirements, complex business logic

### **3. What is CQRS and when should you use it?**

**Answer:**
- **Definition**: Command Query Responsibility Segregation
- **Benefits**: Optimized read/write models, scalability, flexibility
- **Use Cases**: High-read systems, complex business logic, different data models
- **Challenges**: Complexity, eventual consistency, data synchronization
- **Implementation**: Separate command and query handlers, event sourcing

### **4. How do you handle event ordering and consistency?**

**Answer:**
- **Event Ordering**: Use timestamps, sequence numbers, or causal ordering
- **Consistency**: Eventual consistency, strong consistency for critical operations
- **Idempotency**: Handle duplicate events gracefully
- **Saga Pattern**: Manage distributed transactions
- **Compensation**: Handle failures and rollbacks

### **5. What are the challenges of event-driven architecture?**

**Answer:**
- **Complexity**: More complex than request-response patterns
- **Debugging**: Harder to trace and debug event flows
- **Testing**: More difficult to test event interactions
- **Performance**: Potential latency and throughput issues
- **Monitoring**: Need comprehensive event monitoring and alerting

---

**🎉 Event-driven architecture enables building scalable, resilient, and flexible systems!**
