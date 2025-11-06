---
# Auto-generated front matter
Title: Round1-Coding
LastUpdated: 2025-11-06T20:45:58.496210
Tags: []
Status: draft
---

# Atlassian Round 1: Coding Interview

## 📋 Interview Format

- **Duration**: 45-60 minutes
- **Format**: Live coding on shared editor (CoderPad, HackerRank)
- **Language**: Golang preferred, but flexible
- **Focus**: Problem-solving, code quality, and communication

## 🎯 What Atlassian Looks For

### Technical Skills

- **Clean Code**: Readable, maintainable, well-structured
- **Algorithm Efficiency**: Time and space complexity awareness
- **Golang Best Practices**: Idiomatic Go code, proper error handling
- **Testing Mindset**: Ability to think about edge cases and test scenarios

### Soft Skills

- **Communication**: Explain your thought process clearly
- **Problem Decomposition**: Break down complex problems
- **Collaboration**: Work with interviewer as a pair programming partner

## 🔥 Common Problem Categories

### 1. String Manipulation

**Focus**: Atlassian deals with text processing (Jira, Confluence)

#### Example: Text Processing for Jira

```go
// Problem: Parse Jira ticket references from text
// Input: "Please fix BUG-123 and FEATURE-456 in the next sprint"
// Output: ["BUG-123", "FEATURE-456"]

func extractJiraTickets(text string) []string {
    var tickets []string
    re := regexp.MustCompile(`[A-Z]+-\d+`)
    matches := re.FindAllString(text, -1)

    for _, match := range matches {
        if isValidJiraTicket(match) {
            tickets = append(tickets, match)
        }
    }

    return tickets
}

func isValidJiraTicket(ticket string) bool {
    parts := strings.Split(ticket, "-")
    if len(parts) != 2 {
        return false
    }

    // Check if project key is valid (2-10 uppercase letters)
    projectKey := parts[0]
    if len(projectKey) < 2 || len(projectKey) > 10 {
        return false
    }

    for _, char := range projectKey {
        if char < 'A' || char > 'Z' {
            return false
        }
    }

    // Check if ticket number is valid
    ticketNum, err := strconv.Atoi(parts[1])
    return err == nil && ticketNum > 0
}
```

#### Example: Markdown Parser

```go
// Problem: Parse markdown headers and create table of contents
// Input: "# Header 1\n## Header 2\n### Header 3"
// Output: [{"level": 1, "text": "Header 1"}, {"level": 2, "text": "Header 2"}]

type Header struct {
    Level int
    Text  string
}

func parseMarkdownHeaders(content string) []Header {
    var headers []Header
    lines := strings.Split(content, "\n")

    for _, line := range lines {
        if header := parseHeader(line); header != nil {
            headers = append(headers, *header)
        }
    }

    return headers
}

func parseHeader(line string) *Header {
    trimmed := strings.TrimSpace(line)
    if !strings.HasPrefix(trimmed, "#") {
        return nil
    }

    level := 0
    for i, char := range trimmed {
        if char == '#' {
            level++
        } else {
            break
        }
    }

    if level == 0 || level > 6 {
        return nil
    }

    text := strings.TrimSpace(trimmed[level:])
    if text == "" {
        return nil
    }

    return &Header{Level: level, Text: text}
}
```

### 2. Data Structures and Algorithms

**Focus**: Efficient data processing for large-scale systems

#### Example: Task Dependency Resolution

```go
// Problem: Resolve task dependencies in correct order
// Input: [{"task": "A", "deps": ["B", "C"]}, {"task": "B", "deps": ["C"]}, {"task": "C", "deps": []}]
// Output: ["C", "B", "A"] (topological sort)

type Task struct {
    Name string
    Deps []string
}

func resolveTaskDependencies(tasks []Task) ([]string, error) {
    // Build dependency graph
    graph := make(map[string][]string)
    inDegree := make(map[string]int)

    for _, task := range tasks {
        graph[task.Name] = task.Deps
        inDegree[task.Name] = len(task.Deps)
    }

    // Topological sort using Kahn's algorithm
    var queue []string
    var result []string

    // Find tasks with no dependencies
    for task, degree := range inDegree {
        if degree == 0 {
            queue = append(queue, task)
        }
    }

    for len(queue) > 0 {
        current := queue[0]
        queue = queue[1:]
        result = append(result, current)

        // Reduce in-degree for dependent tasks
        for task, deps := range graph {
            for _, dep := range deps {
                if dep == current {
                    inDegree[task]--
                    if inDegree[task] == 0 {
                        queue = append(queue, task)
                    }
                }
            }
        }
    }

    // Check for circular dependencies
    if len(result) != len(tasks) {
        return nil, errors.New("circular dependency detected")
    }

    return result, nil
}
```

#### Example: Efficient Search in Large Dataset

```go
// Problem: Implement autocomplete for project names
// Input: ["Atlassian", "Atlas", "Atmosphere", "Atom"] + query "At"
// Output: ["Atlassian", "Atlas", "Atmosphere"]

type TrieNode struct {
    children map[rune]*TrieNode
    isEnd    bool
    word     string
}

type AutocompleteService struct {
    root *TrieNode
}

func NewAutocompleteService() *AutocompleteService {
    return &AutocompleteService{
        root: &TrieNode{children: make(map[rune]*TrieNode)},
    }
}

func (as *AutocompleteService) Insert(word string) {
    node := as.root
    for _, char := range word {
        if node.children[char] == nil {
            node.children[char] = &TrieNode{children: make(map[rune]*TrieNode)}
        }
        node = node.children[char]
    }
    node.isEnd = true
    node.word = word
}

func (as *AutocompleteService) Search(prefix string) []string {
    node := as.root

    // Navigate to prefix node
    for _, char := range prefix {
        if node.children[char] == nil {
            return []string{}
        }
        node = node.children[char]
    }

    // Collect all words with this prefix
    var results []string
    as.collectWords(node, &results)
    return results
}

func (as *AutocompleteService) collectWords(node *TrieNode, results *[]string) {
    if node.isEnd {
        *results = append(*results, node.word)
    }

    for _, child := range node.children {
        as.collectWords(child, results)
    }
}
```

### 3. Concurrency and Goroutines

**Focus**: Atlassian systems handle high concurrency

#### Example: Rate Limiter

```go
// Problem: Implement rate limiter for API endpoints
// Requirements: Allow 100 requests per minute per user

type RateLimiter struct {
    requests map[string][]time.Time
    limit    int
    window   time.Duration
    mutex    sync.RWMutex
}

func NewRateLimiter(limit int, window time.Duration) *RateLimiter {
    return &RateLimiter{
        requests: make(map[string][]time.Time),
        limit:    limit,
        window:   window,
    }
}

func (rl *RateLimiter) Allow(userID string) bool {
    rl.mutex.Lock()
    defer rl.mutex.Unlock()

    now := time.Now()
    cutoff := now.Add(-rl.window)

    // Clean old requests
    requests := rl.requests[userID]
    var validRequests []time.Time
    for _, reqTime := range requests {
        if reqTime.After(cutoff) {
            validRequests = append(validRequests, reqTime)
        }
    }

    if len(validRequests) >= rl.limit {
        return false
    }

    validRequests = append(validRequests, now)
    rl.requests[userID] = validRequests

    return true
}
```

#### Example: Worker Pool for Background Tasks

```go
// Problem: Process background tasks (emails, notifications) efficiently

type Task struct {
    ID   string
    Data interface{}
    Type string
}

type WorkerPool struct {
    workers    int
    jobQueue   chan Task
    resultQueue chan Task
    wg         sync.WaitGroup
}

func NewWorkerPool(workers int) *WorkerPool {
    return &WorkerPool{
        workers:     workers,
        jobQueue:    make(chan Task, 100),
        resultQueue: make(chan Task, 100),
    }
}

func (wp *WorkerPool) Start() {
    for i := 0; i < wp.workers; i++ {
        wp.wg.Add(1)
        go wp.worker(i)
    }
}

func (wp *WorkerPool) worker(id int) {
    defer wp.wg.Done()

    for task := range wp.jobQueue {
        // Process task based on type
        switch task.Type {
        case "email":
            wp.processEmail(task)
        case "notification":
            wp.processNotification(task)
        default:
            log.Printf("Unknown task type: %s", task.Type)
        }

        wp.resultQueue <- task
    }
}

func (wp *WorkerPool) Submit(task Task) {
    wp.jobQueue <- task
}

func (wp *WorkerPool) Close() {
    close(wp.jobQueue)
    wp.wg.Wait()
    close(wp.resultQueue)
}
```

### 4. System Integration

**Focus**: Atlassian products integrate with many external systems

#### Example: Webhook Handler

```go
// Problem: Handle incoming webhooks from external systems

type WebhookHandler struct {
    validators map[string]func([]byte) error
    processors map[string]func(WebhookPayload) error
    mutex      sync.RWMutex
}

type WebhookPayload struct {
    Source    string      `json:"source"`
    EventType string      `json:"event_type"`
    Data      interface{} `json:"data"`
    Timestamp time.Time   `json:"timestamp"`
}

func NewWebhookHandler() *WebhookHandler {
    return &WebhookHandler{
        validators: make(map[string]func([]byte) error),
        processors: make(map[string]func(WebhookPayload) error),
    }
}

func (wh *WebhookHandler) RegisterValidator(source string, validator func([]byte) error) {
    wh.mutex.Lock()
    defer wh.mutex.Unlock()
    wh.validators[source] = validator
}

func (wh *WebhookHandler) RegisterProcessor(eventType string, processor func(WebhookPayload) error) {
    wh.mutex.Lock()
    defer wh.mutex.Unlock()
    wh.processors[eventType] = processor
}

func (wh *WebhookHandler) HandleWebhook(source string, payload []byte) error {
    wh.mutex.RLock()
    validator, exists := wh.validators[source]
    wh.mutex.RUnlock()

    if !exists {
        return errors.New("unknown source")
    }

    if err := validator(payload); err != nil {
        return fmt.Errorf("validation failed: %w", err)
    }

    var webhookPayload WebhookPayload
    if err := json.Unmarshal(payload, &webhookPayload); err != nil {
        return fmt.Errorf("unmarshal failed: %w", err)
    }

    wh.mutex.RLock()
    processor, exists := wh.processors[webhookPayload.EventType]
    wh.mutex.RUnlock()

    if !exists {
        return errors.New("unknown event type")
    }

    return processor(webhookPayload)
}
```

## 🧪 Testing Patterns

### Unit Testing

```go
func TestExtractJiraTickets(t *testing.T) {
    tests := []struct {
        name     string
        input    string
        expected []string
    }{
        {
            name:     "valid tickets",
            input:    "Please fix BUG-123 and FEATURE-456",
            expected: []string{"BUG-123", "FEATURE-456"},
        },
        {
            name:     "no tickets",
            input:    "This is just regular text",
            expected: []string{},
        },
        {
            name:     "invalid format",
            input:    "Fix bug-123 and FEATURE-",
            expected: []string{},
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := extractJiraTickets(tt.input)
            if !slicesEqual(result, tt.expected) {
                t.Errorf("extractJiraTickets() = %v, want %v", result, tt.expected)
            }
        })
    }
}
```

### Integration Testing

```go
func TestWebhookHandler(t *testing.T) {
    handler := NewWebhookHandler()

    // Register test validator and processor
    handler.RegisterValidator("test", func(payload []byte) error {
        var data map[string]interface{}
        return json.Unmarshal(payload, &data)
    })

    handler.RegisterProcessor("test_event", func(payload WebhookPayload) error {
        if payload.Source != "test" {
            return errors.New("unexpected source")
        }
        return nil
    })

    // Test valid webhook
    validPayload := `{"source": "test", "event_type": "test_event", "data": {}, "timestamp": "2023-01-01T00:00:00Z"}`
    err := handler.HandleWebhook("test", []byte(validPayload))
    if err != nil {
        t.Errorf("HandleWebhook() error = %v", err)
    }
}
```

## 🎯 Interview Tips

### Before the Interview

1. **Practice Golang**: Focus on idiomatic Go patterns
2. **Review Atlassian Products**: Understand Jira, Confluence, Bitbucket
3. **Prepare Examples**: Have 2-3 projects ready to discuss
4. **Study System Design**: Basic understanding of scalability

### During the Interview

1. **Clarify Requirements**: Ask about edge cases and constraints
2. **Think Out Loud**: Explain your approach before coding
3. **Start Simple**: Get basic solution working first
4. **Iterate**: Improve time/space complexity step by step
5. **Test Your Code**: Walk through test cases

### Common Pitfalls to Avoid

1. **Silent Coding**: Always explain your thought process
2. **Premature Optimization**: Get it working first, then optimize
3. **Ignoring Edge Cases**: Ask about null inputs, empty strings, etc.
4. **Poor Error Handling**: Always handle errors properly in Go
5. **Not Testing**: Always test your solution with examples

## 📚 Preparation Resources

### Golang Resources

- [Effective Go](https://golang.org/doc/effective_go.html/)
- [Go by Example](https://gobyexample.com/)
- [Golang Patterns](../shared/golang-coding-patterns.md)

### Algorithm Practice

- [LeetCode](https://leetcode.com/) - Focus on medium difficulty
- [HackerRank](https://www.hackerrank.com/) - Golang track
- [DSA Questions](../shared/dsa-questions.md)

### Atlassian-Specific

- [Atlassian Developer Documentation](https://developer.atlassian.com/)
- [Jira REST API](https://developer.atlassian.com/cloud/jira/platform/rest/v2/)
- [Confluence REST API](https://developer.atlassian.com/cloud/confluence/rest/v1/)

## 🔗 Related Content

- [System Design Patterns](../shared/system-design-patterns.md) - For Round 2
- [Behavioral Questions](../shared/behavioral-bank.md) - For Round 3
- [Golang Coding Patterns](../shared/golang-coding-patterns.md) - Deep dive into Go
