# 🛠️ Interview Preparation Tools & Utilities

> **Comprehensive collection of tools, scripts, and utilities for interview preparation**

## 📚 Table of Contents

1. [Coding Practice Tools](#-coding-practice-tools)
2. [System Design Tools](#-system-design-tools)
3. [Behavioral Interview Tools](#-behavioral-interview-tools)
4. [Study Planning Tools](#-study-planning-tools)
5. [Mock Interview Tools](#-mock-interview-tools)
6. [Progress Tracking Tools](#-progress-tracking-tools)
7. [Resource Management Tools](#-resource-management-tools)

---

## 💻 Coding Practice Tools

### LeetCode Practice Tracker

```python
#!/usr/bin/env python3
"""
LeetCode Practice Tracker
Tracks solved problems, difficulty levels, and progress
"""

import json
import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

@dataclass
class LeetCodeProblem:
    id: int
    title: str
    difficulty: str  # Easy, Medium, Hard
    category: str
    solved_date: str
    time_taken: int  # minutes
    solution_quality: int  # 1-5
    notes: str
    tags: List[str]

class LeetCodeTracker:
    def __init__(self, data_file: str = "leetcode_progress.json"):
        self.data_file = data_file
        self.problems = self.load_problems()
    
    def load_problems(self) -> List[LeetCodeProblem]:
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                return [LeetCodeProblem(**problem) for problem in data]
        except FileNotFoundError:
            return []
    
    def save_problems(self):
        with open(self.data_file, 'w') as f:
            data = [asdict(problem) for problem in self.problems]
            json.dump(data, f, indent=2)
    
    def add_problem(self, problem: LeetCodeProblem):
        self.problems.append(problem)
        self.save_problems()
    
    def get_progress_stats(self) -> Dict:
        total = len(self.problems)
        if total == 0:
            return {"total": 0, "easy": 0, "medium": 0, "hard": 0}
        
        easy = len([p for p in self.problems if p.difficulty == "Easy"])
        medium = len([p for p in self.problems if p.difficulty == "Medium"])
        hard = len([p for p in self.problems if p.difficulty == "Hard"])
        
        return {
            "total": total,
            "easy": easy,
            "medium": medium,
            "hard": hard,
            "easy_percentage": (easy / total) * 100,
            "medium_percentage": (medium / total) * 100,
            "hard_percentage": (hard / total) * 100
        }
    
    def get_category_stats(self) -> Dict:
        categories = {}
        for problem in self.problems:
            if problem.category not in categories:
                categories[problem.category] = 0
            categories[problem.category] += 1
        return categories
    
    def get_weak_areas(self) -> List[str]:
        """Identify categories with fewer solved problems"""
        category_stats = self.get_category_stats()
        if not category_stats:
            return []
        
        avg_problems = sum(category_stats.values()) / len(category_stats)
        weak_areas = [cat for cat, count in category_stats.items() if count < avg_problems * 0.7]
        return weak_areas

# Usage example
if __name__ == "__main__":
    tracker = LeetCodeTracker()
    
    # Add a solved problem
    problem = LeetCodeProblem(
        id=1,
        title="Two Sum",
        difficulty="Easy",
        category="Array",
        solved_date="2024-01-15",
        time_taken=15,
        solution_quality=4,
        notes="Used hash map approach",
        tags=["hash-table", "array"]
    )
    
    tracker.add_problem(problem)
    print("Progress Stats:", tracker.get_progress_stats())
    print("Category Stats:", tracker.get_category_stats())
    print("Weak Areas:", tracker.get_weak_areas())
```

### Coding Problem Generator

```python
#!/usr/bin/env python3
"""
Coding Problem Generator
Generates random coding problems for practice
"""

import random
import json
from typing import List, Dict

class ProblemGenerator:
    def __init__(self, problems_file: str = "coding_problems.json"):
        self.problems = self.load_problems(problems_file)
    
    def load_problems(self, file_path: str) -> List[Dict]:
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.get_default_problems()
    
    def get_default_problems(self) -> List[Dict]:
        return [
            {
                "id": 1,
                "title": "Two Sum",
                "difficulty": "Easy",
                "category": "Array",
                "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
                "examples": [
                    {"input": "nums = [2,7,11,15], target = 9", "output": "[0,1]"},
                    {"input": "nums = [3,2,4], target = 6", "output": "[1,2]"}
                ],
                "constraints": ["2 <= nums.length <= 10^4", "-10^9 <= nums[i] <= 10^9"],
                "time_complexity": "O(n)",
                "space_complexity": "O(n)"
            },
            {
                "id": 2,
                "title": "Add Two Numbers",
                "difficulty": "Medium",
                "category": "Linked List",
                "description": "You are given two non-empty linked lists representing two non-negative integers. Add the two numbers and return the sum as a linked list.",
                "examples": [
                    {"input": "l1 = [2,4,3], l2 = [5,6,4]", "output": "[7,0,8]"},
                    {"input": "l1 = [0], l2 = [0]", "output": "[0]"}
                ],
                "constraints": ["The number of nodes in each linked list is in the range [1, 100]"],
                "time_complexity": "O(max(m,n))",
                "space_complexity": "O(max(m,n))"
            }
        ]
    
    def get_random_problem(self, difficulty: str = None, category: str = None) -> Dict:
        filtered_problems = self.problems
        
        if difficulty:
            filtered_problems = [p for p in filtered_problems if p["difficulty"] == difficulty]
        
        if category:
            filtered_problems = [p for p in filtered_problems if p["category"] == category]
        
        if not filtered_problems:
            return None
        
        return random.choice(filtered_problems)
    
    def get_problem_by_id(self, problem_id: int) -> Dict:
        for problem in self.problems:
            if problem["id"] == problem_id:
                return problem
        return None
    
    def get_problems_by_category(self, category: str) -> List[Dict]:
        return [p for p in self.problems if p["category"] == category]
    
    def get_problems_by_difficulty(self, difficulty: str) -> List[Dict]:
        return [p for p in self.problems if p["difficulty"] == difficulty]

# Usage example
if __name__ == "__main__":
    generator = ProblemGenerator()
    
    # Get random easy problem
    problem = generator.get_random_problem(difficulty="Easy")
    if problem:
        print(f"Problem: {problem['title']}")
        print(f"Difficulty: {problem['difficulty']}")
        print(f"Description: {problem['description']}")
```

### Code Review Checklist

```markdown
# Code Review Checklist

## Functionality
- [ ] Code works as expected
- [ ] Edge cases are handled
- [ ] Error handling is appropriate
- [ ] Input validation is present
- [ ] Output format is correct

## Code Quality
- [ ] Code is readable and well-structured
- [ ] Variable names are descriptive
- [ ] Functions are focused and single-purpose
- [ ] Code follows style guidelines
- [ ] No code duplication

## Performance
- [ ] Time complexity is optimal
- [ ] Space complexity is reasonable
- [ ] No unnecessary computations
- [ ] Efficient data structures used
- [ ] Memory leaks avoided

## Testing
- [ ] Unit tests are present
- [ ] Test cases cover edge cases
- [ ] Tests are meaningful and not trivial
- [ ] Test coverage is adequate
- [ ] Tests are maintainable

## Security
- [ ] Input sanitization is present
- [ ] No sensitive data exposure
- [ ] Proper authentication/authorization
- [ ] SQL injection prevention
- [ ] XSS prevention

## Documentation
- [ ] Code is self-documenting
- [ ] Complex logic is commented
- [ ] API documentation is present
- [ ] README is updated
- [ ] Changelog is updated
```

---

## 🏗️ System Design Tools

### System Design Template

```markdown
# System Design Template

## 1. Requirements Gathering
### Functional Requirements
- [ ] Core functionality
- [ ] User interactions
- [ ] Data operations
- [ ] Integration points

### Non-Functional Requirements
- [ ] Performance (latency, throughput)
- [ ] Scalability (users, data, requests)
- [ ] Availability (uptime, reliability)
- [ ] Security (authentication, authorization)
- [ ] Consistency (data consistency model)

### Constraints
- [ ] Budget constraints
- [ ] Time constraints
- [ ] Technology constraints
- [ ] Regulatory constraints

## 2. Capacity Estimation
### Scale
- [ ] Number of users
- [ ] Requests per second
- [ ] Data volume
- [ ] Storage requirements

### Calculations
- [ ] Daily active users
- [ ] Peak traffic
- [ ] Data growth rate
- [ ] Bandwidth requirements

## 3. High-Level Design
### Components
- [ ] Client applications
- [ ] Load balancers
- [ ] Application servers
- [ ] Databases
- [ ] Caching layers
- [ ] CDN

### Data Flow
- [ ] User request flow
- [ ] Data processing flow
- [ ] Response flow
- [ ] Error handling flow

## 4. Detailed Design
### Database Design
- [ ] Data models
- [ ] Relationships
- [ ] Indexing strategy
- [ ] Sharding strategy
- [ ] Replication strategy

### API Design
- [ ] REST endpoints
- [ ] Request/response formats
- [ ] Authentication
- [ ] Rate limiting
- [ ] Versioning

### Caching Strategy
- [ ] Cache layers
- [ ] Cache invalidation
- [ ] Cache consistency
- [ ] Cache warming

## 5. Scalability
### Horizontal Scaling
- [ ] Load balancing
- [ ] Database sharding
- [ ] Microservices
- [ ] Container orchestration

### Performance Optimization
- [ ] Caching
- [ ] Database optimization
- [ ] CDN usage
- [ ] Compression

## 6. Reliability
### Fault Tolerance
- [ ] Circuit breakers
- [ ] Retry mechanisms
- [ ] Timeout handling
- [ ] Graceful degradation

### Monitoring
- [ ] Health checks
- [ ] Metrics collection
- [ ] Logging
- [ ] Alerting

## 7. Security
### Authentication
- [ ] User authentication
- [ ] API authentication
- [ ] Multi-factor authentication
- [ ] Single sign-on

### Authorization
- [ ] Role-based access control
- [ ] Permission management
- [ ] Resource access control
- [ ] Audit logging

## 8. Trade-offs
### Performance vs Consistency
- [ ] CAP theorem considerations
- [ ] Consistency models
- [ ] Performance implications
- [ ] Business requirements

### Cost vs Performance
- [ ] Infrastructure costs
- [ ] Performance requirements
- [ ] Optimization strategies
- [ ] ROI considerations
```

### Architecture Diagram Generator

```python
#!/usr/bin/env python3
"""
Architecture Diagram Generator
Generates system architecture diagrams using Mermaid
"""

class ArchitectureDiagramGenerator:
    def __init__(self):
        self.diagram_types = {
            "microservices": self.generate_microservices_diagram,
            "monolith": self.generate_monolith_diagram,
            "event_driven": self.generate_event_driven_diagram,
            "layered": self.generate_layered_diagram
        }
    
    def generate_microservices_diagram(self, services: List[str]) -> str:
        diagram = "graph TB\n"
        diagram += "    Client[Client Applications]\n"
        diagram += "    LB[Load Balancer]\n"
        diagram += "    Gateway[API Gateway]\n"
        
        for service in services:
            diagram += f"    {service}[{service} Service]\n"
        
        diagram += "    DB[(Database)]\n"
        diagram += "    Cache[(Cache)]\n"
        diagram += "    Queue[Message Queue]\n"
        
        diagram += "    Client --> LB\n"
        diagram += "    LB --> Gateway\n"
        diagram += "    Gateway --> " + "\n    Gateway --> ".join(services) + "\n"
        
        for service in services:
            diagram += f"    {service} --> DB\n"
            diagram += f"    {service} --> Cache\n"
            diagram += f"    {service} --> Queue\n"
        
        return diagram
    
    def generate_event_driven_diagram(self) -> str:
        return """
graph TB
    Client[Client Applications]
    API[API Gateway]
    Services[Microservices]
    EventBus[Event Bus]
    EventStore[(Event Store)]
    Projections[Read Models]
    
    Client --> API
    API --> Services
    Services --> EventBus
    EventBus --> EventStore
    EventStore --> Projections
    Projections --> API
    """
    
    def generate_diagram(self, diagram_type: str, **kwargs) -> str:
        if diagram_type in self.diagram_types:
            return self.diagram_types[diagram_type](**kwargs)
        else:
            raise ValueError(f"Unknown diagram type: {diagram_type}")

# Usage example
if __name__ == "__main__":
    generator = ArchitectureDiagramGenerator()
    
    services = ["User", "Payment", "Notification", "Analytics"]
    diagram = generator.generate_microservices_diagram(services)
    print(diagram)
```

---

## 🤝 Behavioral Interview Tools

### STAR Method Template

```markdown
# STAR Method Template

## Situation
**Context**: What was the situation? When and where did it occur?
- [ ] Set the scene
- [ ] Provide background
- [ ] Explain the context
- [ ] Keep it concise (30-60 seconds)

## Task
**Responsibility**: What was your role and responsibility?
- [ ] Explain your specific role
- [ ] Describe what you were responsible for
- [ ] Clarify the objective
- [ ] Show ownership

## Action
**Steps**: What specific actions did you take?
- [ ] Describe what you did
- [ ] Explain your thought process
- [ ] Show problem-solving skills
- [ ] Demonstrate leadership
- [ ] Use "I" statements

## Result
**Outcome**: What was the result of your actions?
- [ ] Quantify the impact
- [ ] Show measurable results
- [ ] Explain what you learned
- [ ] Connect to the role you're applying for
```

### Behavioral Question Bank

```python
#!/usr/bin/env python3
"""
Behavioral Question Bank
Collection of common behavioral interview questions
"""

class BehavioralQuestionBank:
    def __init__(self):
        self.questions = {
            "leadership": [
                "Tell me about a time when you had to lead a team through a difficult situation.",
                "Describe a time when you had to motivate a team member who was struggling.",
                "Tell me about a time when you had to make a difficult decision that affected your team.",
                "Describe a situation where you had to lead without formal authority.",
                "Tell me about a time when you had to manage conflicting priorities within your team."
            ],
            "problem_solving": [
                "Tell me about a time when you had to solve a complex problem.",
                "Describe a situation where you had to think outside the box to find a solution.",
                "Tell me about a time when you had to make a decision with incomplete information.",
                "Describe a time when you had to troubleshoot a critical issue.",
                "Tell me about a time when you had to learn something new quickly to solve a problem."
            ],
            "communication": [
                "Tell me about a time when you had to explain a complex technical concept to a non-technical audience.",
                "Describe a situation where you had to deliver bad news to stakeholders.",
                "Tell me about a time when you had to resolve a conflict between team members.",
                "Describe a time when you had to present to senior leadership.",
                "Tell me about a time when you had to give difficult feedback to a colleague."
            ],
            "adaptability": [
                "Tell me about a time when you had to adapt to a major change in your work environment.",
                "Describe a situation where you had to learn a new technology quickly.",
                "Tell me about a time when you had to work with a difficult team member.",
                "Describe a time when you had to pivot your approach due to changing requirements.",
                "Tell me about a time when you had to work under pressure with tight deadlines."
            ],
            "achievement": [
                "Tell me about your greatest professional achievement.",
                "Describe a time when you exceeded expectations on a project.",
                "Tell me about a time when you received recognition for your work.",
                "Describe a situation where you had to overcome significant obstacles to achieve a goal.",
                "Tell me about a time when you had to take initiative to solve a problem."
            ],
            "failure": [
                "Tell me about a time when you failed at something important.",
                "Describe a situation where you made a mistake that had significant consequences.",
                "Tell me about a time when you had to recover from a major setback.",
                "Describe a time when you had to admit you were wrong.",
                "Tell me about a time when you had to learn from a failure."
            ]
        }
    
    def get_questions_by_category(self, category: str) -> List[str]:
        return self.questions.get(category, [])
    
    def get_random_question(self, category: str = None) -> str:
        if category and category in self.questions:
            questions = self.questions[category]
        else:
            all_questions = []
            for cat_questions in self.questions.values():
                all_questions.extend(cat_questions)
            questions = all_questions
        
        import random
        return random.choice(questions)
    
    def get_all_categories(self) -> List[str]:
        return list(self.questions.keys())

# Usage example
if __name__ == "__main__":
    bank = BehavioralQuestionBank()
    
    # Get random leadership question
    question = bank.get_random_question("leadership")
    print(f"Leadership Question: {question}")
    
    # Get all categories
    categories = bank.get_all_categories()
    print(f"Available categories: {categories}")
```

### Interview Practice Tracker

```python
#!/usr/bin/env python3
"""
Interview Practice Tracker
Tracks behavioral interview practice sessions
"""

import json
import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

@dataclass
class PracticeSession:
    date: str
    duration: int  # minutes
    questions_answered: int
    categories: List[str]
    performance_rating: int  # 1-5
    notes: str
    areas_to_improve: List[str]

class InterviewPracticeTracker:
    def __init__(self, data_file: str = "interview_practice.json"):
        self.data_file = data_file
        self.sessions = self.load_sessions()
    
    def load_sessions(self) -> List[PracticeSession]:
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                return [PracticeSession(**session) for session in data]
        except FileNotFoundError:
            return []
    
    def save_sessions(self):
        with open(self.data_file, 'w') as f:
            data = [asdict(session) for session in self.sessions]
            json.dump(data, f, indent=2)
    
    def add_session(self, session: PracticeSession):
        self.sessions.append(session)
        self.save_sessions()
    
    def get_practice_stats(self) -> Dict:
        if not self.sessions:
            return {"total_sessions": 0, "total_time": 0, "avg_rating": 0}
        
        total_sessions = len(self.sessions)
        total_time = sum(session.duration for session in self.sessions)
        avg_rating = sum(session.performance_rating for session in self.sessions) / total_sessions
        
        return {
            "total_sessions": total_sessions,
            "total_time": total_time,
            "avg_rating": round(avg_rating, 2),
            "total_questions": sum(session.questions_answered for session in self.sessions)
        }
    
    def get_improvement_areas(self) -> List[str]:
        improvement_areas = []
        for session in self.sessions:
            improvement_areas.extend(session.areas_to_improve)
        
        # Count frequency of each area
        area_counts = {}
        for area in improvement_areas:
            area_counts[area] = area_counts.get(area, 0) + 1
        
        # Sort by frequency
        sorted_areas = sorted(area_counts.items(), key=lambda x: x[1], reverse=True)
        return [area for area, count in sorted_areas]

# Usage example
if __name__ == "__main__":
    tracker = InterviewPracticeTracker()
    
    # Add a practice session
    session = PracticeSession(
        date="2024-01-15",
        duration=60,
        questions_answered=5,
        categories=["leadership", "problem_solving"],
        performance_rating=4,
        notes="Good examples but need to be more specific",
        areas_to_improve=["quantifying results", "using STAR method"]
    )
    
    tracker.add_session(session)
    print("Practice Stats:", tracker.get_practice_stats())
    print("Improvement Areas:", tracker.get_improvement_areas())
```

---

## 📅 Study Planning Tools

### Study Schedule Generator

```python
#!/usr/bin/env python3
"""
Study Schedule Generator
Creates personalized study schedules for interview preparation
"""

import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class StudySession:
    date: str
    time: str
    duration: int  # minutes
    topic: str
    type: str  # "coding", "system_design", "behavioral"
    priority: int  # 1-5
    completed: bool = False

class StudyScheduleGenerator:
    def __init__(self):
        self.topics = {
            "coding": [
                "Arrays and Strings",
                "Linked Lists",
                "Trees and Graphs",
                "Dynamic Programming",
                "Sorting and Searching",
                "Hash Tables",
                "Stacks and Queues"
            ],
            "system_design": [
                "Scalability",
                "Load Balancing",
                "Caching",
                "Database Design",
                "Microservices",
                "Message Queues",
                "CDN and Edge Computing"
            ],
            "behavioral": [
                "Leadership Examples",
                "Problem Solving",
                "Communication",
                "Adaptability",
                "Achievement Stories",
                "Failure and Learning"
            ]
        }
    
    def generate_schedule(self, 
                         start_date: str, 
                         end_date: str, 
                         hours_per_week: int,
                         focus_areas: List[str] = None) -> List[StudySession]:
        
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        
        sessions = []
        current_date = start
        
        # Calculate sessions per week
        weeks = (end - start).days // 7
        sessions_per_week = (hours_per_week * 60) // 90  # 90-minute sessions
        
        session_id = 0
        while current_date <= end:
            # Generate sessions for this week
            for i in range(sessions_per_week):
                if current_date > end:
                    break
                
                # Select topic based on focus areas
                topic_type = self.select_topic_type(focus_areas, session_id)
                topic = self.select_topic(topic_type, session_id)
                
                session = StudySession(
                    date=current_date.strftime("%Y-%m-%d"),
                    time=f"{9 + (i * 2)}:00",  # 9 AM, 11 AM, 1 PM, etc.
                    duration=90,
                    topic=topic,
                    type=topic_type,
                    priority=self.calculate_priority(topic_type, session_id)
                )
                
                sessions.append(session)
                session_id += 1
            
            current_date += datetime.timedelta(days=7)
        
        return sessions
    
    def select_topic_type(self, focus_areas: List[str], session_id: int) -> str:
        if focus_areas:
            return focus_areas[session_id % len(focus_areas)]
        
        # Default rotation: coding, system_design, behavioral
        types = ["coding", "system_design", "behavioral"]
        return types[session_id % len(types)]
    
    def select_topic(self, topic_type: str, session_id: int) -> str:
        topics = self.topics.get(topic_type, [])
        if not topics:
            return "General Practice"
        
        return topics[session_id % len(topics)]
    
    def calculate_priority(self, topic_type: str, session_id: int) -> int:
        # Higher priority for earlier sessions
        base_priority = 5 - (session_id // 10)
        return max(1, min(5, base_priority))

# Usage example
if __name__ == "__main__":
    generator = StudyScheduleGenerator()
    
    schedule = generator.generate_schedule(
        start_date="2024-01-15",
        end_date="2024-02-15",
        hours_per_week=15,
        focus_areas=["coding", "system_design"]
    )
    
    for session in schedule[:5]:  # Show first 5 sessions
        print(f"{session.date} {session.time}: {session.topic} ({session.type})")
```

### Progress Tracker

```python
#!/usr/bin/env python3
"""
Progress Tracker
Tracks study progress and identifies areas for improvement
"""

import json
import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

@dataclass
class ProgressEntry:
    date: str
    topic: str
    type: str
    time_spent: int  # minutes
    difficulty: int  # 1-5
    confidence: int  # 1-5
    notes: str

class ProgressTracker:
    def __init__(self, data_file: str = "progress.json"):
        self.data_file = data_file
        self.entries = self.load_entries()
    
    def load_entries(self) -> List[ProgressEntry]:
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                return [ProgressEntry(**entry) for entry in data]
        except FileNotFoundError:
            return []
    
    def save_entries(self):
        with open(self.data_file, 'w') as f:
            data = [asdict(entry) for entry in self.entries]
            json.dump(data, f, indent=2)
    
    def add_entry(self, entry: ProgressEntry):
        self.entries.append(entry)
        self.save_entries()
    
    def get_progress_summary(self) -> Dict:
        if not self.entries:
            return {"total_time": 0, "topics_covered": 0, "avg_confidence": 0}
        
        total_time = sum(entry.time_spent for entry in self.entries)
        topics_covered = len(set(entry.topic for entry in self.entries))
        avg_confidence = sum(entry.confidence for entry in self.entries) / len(self.entries)
        
        return {
            "total_time": total_time,
            "topics_covered": topics_covered,
            "avg_confidence": round(avg_confidence, 2),
            "total_sessions": len(self.entries)
        }
    
    def get_weak_areas(self) -> List[Dict]:
        """Identify topics with low confidence scores"""
        topic_stats = {}
        
        for entry in self.entries:
            if entry.topic not in topic_stats:
                topic_stats[entry.topic] = {"total_confidence": 0, "count": 0}
            
            topic_stats[entry.topic]["total_confidence"] += entry.confidence
            topic_stats[entry.topic]["count"] += 1
        
        weak_areas = []
        for topic, stats in topic_stats.items():
            avg_confidence = stats["total_confidence"] / stats["count"]
            if avg_confidence < 3:  # Below average confidence
                weak_areas.append({
                    "topic": topic,
                    "avg_confidence": round(avg_confidence, 2),
                    "sessions": stats["count"]
                })
        
        return sorted(weak_areas, key=lambda x: x["avg_confidence"])
    
    def get_study_recommendations(self) -> List[str]:
        weak_areas = self.get_weak_areas()
        recommendations = []
        
        for area in weak_areas[:3]:  # Top 3 weak areas
            recommendations.append(f"Focus more on {area['topic']} (confidence: {area['avg_confidence']}/5)")
        
        return recommendations

# Usage example
if __name__ == "__main__":
    tracker = ProgressTracker()
    
    # Add a progress entry
    entry = ProgressEntry(
        date="2024-01-15",
        topic="Dynamic Programming",
        type="coding",
        time_spent=120,
        difficulty=4,
        confidence=3,
        notes="Struggled with state transitions"
    )
    
    tracker.add_entry(entry)
    print("Progress Summary:", tracker.get_progress_summary())
    print("Weak Areas:", tracker.get_weak_areas())
    print("Recommendations:", tracker.get_study_recommendations())
```

---

## 🎯 Mock Interview Tools

### Mock Interview Scheduler

```python
#!/usr/bin/env python3
"""
Mock Interview Scheduler
Schedules and manages mock interview sessions
"""

import json
import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

@dataclass
class MockInterview:
    id: str
    date: str
    time: str
    duration: int  # minutes
    type: str  # "coding", "system_design", "behavioral", "mixed"
    interviewer: str
    topics: List[str]
    status: str  # "scheduled", "completed", "cancelled"
    feedback: Optional[str] = None
    rating: Optional[int] = None

class MockInterviewScheduler:
    def __init__(self, data_file: str = "mock_interviews.json"):
        self.data_file = data_file
        self.interviews = self.load_interviews()
    
    def load_interviews(self) -> List[MockInterview]:
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                return [MockInterview(**interview) for interview in data]
        except FileNotFoundError:
            return []
    
    def save_interviews(self):
        with open(self.data_file, 'w') as f:
            data = [asdict(interview) for interview in self.interviews]
            json.dump(data, f, indent=2)
    
    def schedule_interview(self, interview: MockInterview):
        self.interviews.append(interview)
        self.save_interviews()
    
    def get_upcoming_interviews(self) -> List[MockInterview]:
        today = datetime.datetime.now().date()
        upcoming = []
        
        for interview in self.interviews:
            interview_date = datetime.datetime.strptime(interview.date, "%Y-%m-%d").date()
            if interview_date >= today and interview.status == "scheduled":
                upcoming.append(interview)
        
        return sorted(upcoming, key=lambda x: (x.date, x.time))
    
    def get_interview_stats(self) -> Dict:
        total = len(self.interviews)
        completed = len([i for i in self.interviews if i.status == "completed"])
        scheduled = len([i for i in self.interviews if i.status == "scheduled"])
        
        if completed == 0:
            avg_rating = 0
        else:
            ratings = [i.rating for i in self.interviews if i.rating is not None]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        return {
            "total_interviews": total,
            "completed": completed,
            "scheduled": scheduled,
            "avg_rating": round(avg_rating, 2)
        }

# Usage example
if __name__ == "__main__":
    scheduler = MockInterviewScheduler()
    
    # Schedule a mock interview
    interview = MockInterview(
        id="mock_001",
        date="2024-01-20",
        time="14:00",
        duration=60,
        type="coding",
        interviewer="John Doe",
        topics=["Arrays", "Dynamic Programming"],
        status="scheduled"
    )
    
    scheduler.schedule_interview(interview)
    print("Upcoming Interviews:", scheduler.get_upcoming_interviews())
    print("Interview Stats:", scheduler.get_interview_stats())
```

### Interview Feedback Template

```markdown
# Mock Interview Feedback Template

## Interview Details
- **Date**: [Date]
- **Duration**: [Duration]
- **Type**: [Coding/System Design/Behavioral]
- **Interviewer**: [Name]
- **Topics Covered**: [List topics]

## Overall Performance
- **Rating**: [1-5]
- **Strengths**: [List strengths]
- **Areas for Improvement**: [List areas to improve]

## Technical Performance
### Coding (if applicable)
- **Problem Understanding**: [1-5]
- **Solution Approach**: [1-5]
- **Code Quality**: [1-5]
- **Testing**: [1-5]
- **Communication**: [1-5]

### System Design (if applicable)
- **Requirements Gathering**: [1-5]
- **High-Level Design**: [1-5]
- **Detailed Design**: [1-5]
- **Scalability**: [1-5]
- **Trade-offs**: [1-5]

### Behavioral (if applicable)
- **STAR Method**: [1-5]
- **Specificity**: [1-5]
- **Leadership**: [1-5]
- **Problem Solving**: [1-5]
- **Communication**: [1-5]

## Specific Feedback
### What Went Well
- [Specific positive feedback]

### Areas to Improve
- [Specific areas for improvement]

### Action Items
- [ ] [Specific action item 1]
- [ ] [Specific action item 2]
- [ ] [Specific action item 3]

## Next Steps
- [Recommended next steps]
- [Resources to study]
- [Practice recommendations]

## Overall Comments
[Additional comments and recommendations]
```

---

## 📊 Resource Management Tools

### Resource Tracker

```python
#!/usr/bin/env python3
"""
Resource Tracker
Tracks and manages study resources
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

@dataclass
class StudyResource:
    id: str
    title: str
    type: str  # "book", "article", "video", "course", "practice"
    category: str
    url: Optional[str] = None
    status: str = "not_started"  # "not_started", "in_progress", "completed"
    rating: Optional[int] = None
    notes: str = ""

class ResourceTracker:
    def __init__(self, data_file: str = "resources.json"):
        self.data_file = data_file
        self.resources = self.load_resources()
    
    def load_resources(self) -> List[StudyResource]:
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                return [StudyResource(**resource) for resource in data]
        except FileNotFoundError:
            return self.get_default_resources()
    
    def get_default_resources(self) -> List[StudyResource]:
        return [
            StudyResource(
                id="res_001",
                title="Cracking the Coding Interview",
                type="book",
                category="coding",
                status="not_started"
            ),
            StudyResource(
                id="res_002",
                title="System Design Interview",
                type="book",
                category="system_design",
                status="not_started"
            ),
            StudyResource(
                id="res_003",
                title="LeetCode",
                type="practice",
                category="coding",
                url="https://leetcode.com",
                status="not_started"
            )
        ]
    
    def save_resources(self):
        with open(self.data_file, 'w') as f:
            data = [asdict(resource) for resource in self.resources]
            json.dump(data, f, indent=2)
    
    def add_resource(self, resource: StudyResource):
        self.resources.append(resource)
        self.save_resources()
    
    def get_resources_by_category(self, category: str) -> List[StudyResource]:
        return [r for r in self.resources if r.category == category]
    
    def get_resources_by_status(self, status: str) -> List[StudyResource]:
        return [r for r in self.resources if r.status == status]
    
    def update_resource_status(self, resource_id: str, status: str):
        for resource in self.resources:
            if resource.id == resource_id:
                resource.status = status
                break
        self.save_resources()
    
    def get_resource_stats(self) -> Dict:
        total = len(self.resources)
        completed = len([r for r in self.resources if r.status == "completed"])
        in_progress = len([r for r in self.resources if r.status == "in_progress"])
        not_started = len([r for r in self.resources if r.status == "not_started"])
        
        return {
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "not_started": not_started,
            "completion_rate": (completed / total) * 100 if total > 0 else 0
        }

# Usage example
if __name__ == "__main__":
    tracker = ResourceTracker()
    
    # Add a new resource
    resource = StudyResource(
        id="res_004",
        title="Grokking the System Design Interview",
        type="course",
        category="system_design",
        url="https://www.educative.io/courses/grokking-the-system-design-interview",
        status="not_started"
    )
    
    tracker.add_resource(resource)
    print("Resource Stats:", tracker.get_resource_stats())
    print("Coding Resources:", tracker.get_resources_by_category("coding"))
```

---

## 🚀 Quick Start Guide

### Setting Up Your Interview Preparation Environment

1. **Install Required Tools**:
   ```bash
   # Python environment
   python3 -m venv interview_prep
   source interview_prep/bin/activate
   pip install -r requirements.txt
   
   # Git for version control
   git init
   git add .
   git commit -m "Initial interview preparation setup"
   ```

2. **Create Your Study Plan**:
   ```python
   from study_planning import StudyScheduleGenerator
   
   generator = StudyScheduleGenerator()
   schedule = generator.generate_schedule(
       start_date="2024-01-15",
       end_date="2024-03-15",
       hours_per_week=20,
       focus_areas=["coding", "system_design", "behavioral"]
   )
   ```

3. **Start Tracking Progress**:
   ```python
   from progress_tracking import ProgressTracker
   
   tracker = ProgressTracker()
   # Add your progress entries as you study
   ```

4. **Schedule Mock Interviews**:
   ```python
   from mock_interviews import MockInterviewScheduler
   
   scheduler = MockInterviewScheduler()
   # Schedule regular mock interviews
   ```

### Daily Workflow

1. **Morning**: Review today's study schedule
2. **Study Session**: Focus on scheduled topics
3. **Progress Update**: Log your progress and confidence
4. **Evening**: Review weak areas and plan tomorrow

### Weekly Review

1. **Progress Analysis**: Review your progress stats
2. **Weak Areas**: Identify topics needing more attention
3. **Schedule Adjustment**: Update your study schedule
4. **Mock Interview**: Schedule a mock interview session

---

**🛠️ Use these tools to maximize your interview preparation effectiveness! Good luck! 🚀**
