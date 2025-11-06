---
# Auto-generated front matter
Title: Readme
LastUpdated: 2025-11-06T20:45:58.372293
Tags: []
Status: draft
---

# Real-Time Notification Fan-out (Prompt 2)

Skeleton service demonstrating multi-channel dispatch abstraction.

## Components

- Dispatcher registry
- Channel implementations (email/sms dummy)
- Retry with exponential backoff (placeholder)

## Run

```bash
cd 06_projects_portfolio/implementations/realtime_notification_fanout
go run ./...
```

## Extend

- Add priority queue
- Add dead-letter for permanent failures

## Testing

- Channel failure injection
- Retry backoff timing
