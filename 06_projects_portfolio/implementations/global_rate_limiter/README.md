---
# Auto-generated front matter
Title: Readme
LastUpdated: 2025-11-06T20:45:58.372558
Tags: []
Status: draft
---

# Global Rate Limiter (Prompt 1)

Implements a distributed rate limiter abstraction (token bucket + leaky bucket hybrid) with pluggable backends. Current skeleton uses in-memory map; extend to Redis / Kafka for multi-region coordination.

## Goals

- Per-tenant + global limits
- Burst absorption with smoothing
- Sliding window analytics for observability

## Architecture (Skeleton)

```text
[Client] -> [Limiter Library] -> [Limiter Core] -> [State Backend]
```
Backend interface lets you swap in Redis cluster. For multi-region, use local region counters + periodic CRDT/Gossip merge.

## Run

```bash
cd 06_projects_portfolio/implementations/global_rate_limiter
go run ./...
```

## Extend

- Add `redis_backend.go` implementing atomic LUA script for token deduction.
- Add metrics (Prometheus) with: remaining_tokens, refill_latency.

## Testing Strategy

- Unit: deterministic refill math
- Concurrency: race detector (`go test -race`)
- Load: k6 script in `tools/benchmark/k6-rate-limiter-script.js`

## Next

- p99 latency budget enforcement
- Adaptive limits based on error rate
