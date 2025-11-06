---
# Auto-generated front matter
Title: Faang Style Prompts
LastUpdated: 2025-11-06T20:45:57.734257
Tags: []
Status: draft
---

# FAANG-Style System Design Prompts

Use the scaffold in `_meta/STYLE_GUIDE.md` for answers. Each prompt includes context + extra constraints for advanced depth.

## Prompts (Expanded to 25)

1. Global Rate Limiter API Edge  
   Multi-region deployment, tenant isolation, burst absorption.
2. Real-Time Notification Fan-out Service  
   50M users, multi-channel (push/email/SMS), prioritization + retries.
3. Payment Gateway & Ledger  
   Idempotency, partial failure recovery, reconciliation window.
4. Multi-Region Config Distribution  
   Read-mostly, versioning, low replication lag, audit history.
5. Feature Flag Platform with Experimentation  
   Targeting rules, exposure logging at scale, real-time invalidation.
6. Streaming Analytics Pipeline  
   Window aggregation, late event handling, enrichment timeouts.
7. Global Chat & Presence System  
   Fan-out vs shard trade-offs, typing indicators, ephemeral state.
8. Document Collaboration (Operational Transform)  
   Conflict resolution, version snapshots, offline reconciliation.
9. Risk Scoring Engine  
   Real-time scoring with feature cache warm strategy, SLA budgeting.
10. Distributed Job Scheduler  
    Cron replacement across services, fairness, backpressure.
11. Real-Time Fraud Detection Pipeline  
    Streaming feature joins, low-latency scoring, model update cadence.
12. Global Search Suggestion Service  
    Prefix index, caching tiers, miss handling, relevance freshness.
13. Media Content Delivery Network Controller  
    Regional edge selection, cache warming, invalidation strategy.
14. Distributed Feature Store  
    Low-latency online reads + batch backfills, schema evolution.
15. Session Management at Scale  
    Sticky vs stateless trade-offs, revocation, rolling upgrade.
16. Real-Time Metrics Aggregator  
    High cardinality handling, downsampling, retention tiers.
17. Order Matching Engine (Simplified)  
    Consistency vs throughput, in-memory book replication.
18. Experiment Analytics Service  
    Exposure, conversion, statistical guardrails, backfill.
19. Global File Storage Metadata Layer  
    Namespace sharding, consistency, version history.
20. Webhooks Delivery Platform  
    Retry scheduling, exponential backoff, signature verification.
21. Personalization Recommendation Pipeline  
    Candidate generation + ranking separation, freshness SLAs.
22. Distributed Cache Invalidation Coordinator  
    Version vectors vs pub/sub, hot key mitigation.
23. API Gateway with Dynamic Routing  
    Canary releases, circuit breaking, integration observability.
24. Real-Time Leaderboard Service  
    Sorted set scaling, partial updates, anti-cheat validation.
25. Data Migration Framework  
    Background chunking, backpressure, consistency audit.

## Advanced Constraints Ideas

- Inject partition outage scenario.
- Enforce read/write consistency trade-off articulation.
- Require capacity estimation (QPS, storage footprint, throughput).
- Introduce cost ceiling forcing architectural simplification.
- Simulate latency budget shrink (e.g., p99 target drops from 300ms to 120ms).

## Evaluation Dimensions

| Dimension | Strong Indicators |
|-----------|-------------------|
| Clarification | Explicit scope boundaries, constraints enumerated |
| Capacity | Concrete QPS, storage, growth assumptions |
| Architecture | Logical layering, clear data/control flows |
| Trade-offs | Latency vs consistency, cost vs reliability justified |
| Failure Handling | Enumerates partial failures + mitigation patterns |
| Evolution | Clear V1 -> V2 scaling path |

## Exemplar Answer Structure (Applied to First 5 in `faang_prompt_answers.md`)

Each answer includes: requirements table, capacity math, ASCII+Mermaid diagram, trade-off matrix, failure scenarios, evolution roadmap, metrics & SLOs.

## Code Execution & Repository References

Below is a mapping from the first 5 prompts to runnable skeleton implementations added under `06_projects_portfolio/implementations/`. These starters focus on core logic; extend them with persistence, networking, observability, and resilience patterns as you iterate.

| Prompt # | Name | Path | Primary Language | Run (Go) | Alt (Node) |
|----------|------|------|------------------|----------|------------|
| 1 | Global Rate Limiter | `06_projects_portfolio/implementations/global_rate_limiter` | Go | `cd 06_projects_portfolio/implementations/global_rate_limiter && go run ./...` | `cd 06_projects_portfolio/implementations/global_rate_limiter/node && npm start` |
| 2 | Real-Time Notification Fan-out | `06_projects_portfolio/implementations/realtime_notification_fanout` | Go | `cd 06_projects_portfolio/implementations/realtime_notification_fanout && go run ./...` | (add Node variant later) |
| 3 | Payment Gateway & Ledger | `06_projects_portfolio/implementations/payment_gateway_ledger` | Go | `cd 06_projects_portfolio/implementations/payment_gateway_ledger && go run ./...` | (planned) |
| 4 | Multi-Region Config Distribution | `06_projects_portfolio/implementations/multi_region_config_distribution` | Go | `cd 06_projects_portfolio/implementations/multi_region_config_distribution && go run ./...` | (planned) |
| 5 | Feature Flag Platform | `06_projects_portfolio/implementations/feature_flag_platform` | Go | `cd 06_projects_portfolio/implementations/feature_flag_platform && go run ./...` | (planned) |

### HTTP API Quick Reference

| Service | Port | Endpoint | Method | Purpose | Sample Curl |
|---------|------|----------|--------|---------|-------------|
| Rate Limiter | 8080 | `/ratelimit/allow?tokens=3` | GET | Consume tokens | `curl -s 'http://localhost:8080/ratelimit/allow?tokens=3'` |
| Payment Gateway | 8081 | `/payments/authorize?user=u1&amount=500&currency=INR&requestId=abc` | GET | Idempotent auth | `curl -s 'http://localhost:8081/payments/authorize?user=u1&amount=500&currency=INR&requestId=abc'` |
| Payment Gateway | 8081 | `/payments/capture?user=u1&amount=500&currency=INR&requestId=abc&authId=AUTH_ID` | GET | Capture funds | `curl -s 'http://localhost:8081/payments/capture?user=u1&amount=500&currency=INR&requestId=abc&authId=AUTH_ID'` |
| Feature Flags | 8082 | `/flags/new_ui` | GET | Read flag state | `curl -s 'http://localhost:8082/flags/new_ui'` |
| Feature Flags | 8082 | `/flags/beta_payment?on=true` | PUT | Update flag | `curl -X PUT -s 'http://localhost:8082/flags/beta_payment?on=true'` |
| Config Dist. | 8083 | `/config/feature_x` | GET | Read config | `curl -s 'http://localhost:8083/config/feature_x'` |
| Config Dist. | 8083 | `/config/feature_x?value=on` | PUT | Write config | `curl -X PUT -s 'http://localhost:8083/config/feature_x?value=on'` |
| Config Audit | 8083 | `/config_audit` | GET | List versions | `curl -s 'http://localhost:8083/config_audit'` |
| Notification Fan-out | 8084 | `/notify/dispatch` | POST | Dispatch message | `curl -s -X POST -H 'Content-Type: application/json' -d '{"userId":"u1","channel":"email","content":"Hello"}' http://localhost:8084/notify/dispatch` |

### Extending Skeletons (Suggested Next Steps)

1. Add HTTP layer (Go: `net/http`, Node: Express/Fastify) exposing endpoints:
    - Rate Limiter: `GET /allow?tenant=...`
    - Notifications: `POST /dispatch`
    - Ledger: `POST /payments` (idempotent)
    - Config: `GET /config/{key}` / `PUT /config/{key}`
    - Feature Flags: `GET /evaluate?flag=x&user=...`
2. Introduce persistence layer (Postgres / Redis) & update README with migration commands.
3. Add metrics (Prometheus exporters) and tracing (OpenTelemetry) instrumentation.
4. Write k6 load scripts per service under `tools/benchmark/` (rate limiter example already scaffolded).
5. Implement resilience patterns: circuit breaker around external calls, backoff/retry for notification channels.
6. Document SLOs in each README (availability, latency, error budget policy).

### Validation Checklist

- Builds succeed (`go build ./...` in each directory).
- Basic workflow exercised (`go run` produces output demonstrating core behavior).
- Idempotency confirmed for payment gateway (replay returns same auth id, only one ledger append).
- Version increment logic correct for config distribution service.
- Feature flag store updates reflect immediately in retrieval.
- Notification dispatcher sends to both channels sequentially.

### Future Additions

| Area | Enhancement |
|------|-------------|
| Persistence | Replace in-memory maps with Postgres + migrations |
| Distribution | Add Redis cluster + pub/sub invalidation (flags, config) |
| Observability | Add `/metrics` endpoint, tracing spans around critical paths |
| Security | Add auth middleware + API keys / JWT |
| Load | Parameterize k6 scripts with env-driven target host |
| CI | Add GitHub Actions to build & run unit tests for each module |

> NOTE: The current Go examples are intentionally minimal. They are teaching artifacts aligned with interview storytelling: start simple, articulate evolution plan.
