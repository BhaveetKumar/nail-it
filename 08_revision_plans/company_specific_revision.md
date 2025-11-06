---
# Auto-generated front matter
Title: Company Specific Revision
LastUpdated: 2025-11-06T20:45:58.328246
Tags: []
Status: draft
---

# Company-Specific Revision Mapping

| Company | Emphasis | Core Drill Topics | Target Artifacts | Metrics Focus |
|---------|----------|-------------------|------------------|---------------|
| FAANG (generic) | Scale + trade-offs | Global rate limiting, multi-region failover, caching tiers | multi_region_failover.md, rate_limiting_tradeoffs.md | p99 latency, availability, cache hit ratio |
| Coinbase | Data integrity & security | Ledger consistency, idempotent workflows, audit & encryption | payment_gateway_service.yaml, threat_modeling_walkthrough.md | double-spend rate, reconciliation diff %, encryption coverage |
| Rippling | Multi-tenant platform | Isolation strategies, config propagation, feature flags | feature_flag_system.yaml, tenancy_isolation.md | config propagation lag, per-tenant isolation tests |
| Fintech (general) | Reliability + compliance | Fraud pipeline, retriable workflows, strong audit trail | payment_pipeline_flow.md, ledger_events_design.md | fraud false positive %, retry success %, audit completeness |
| Productivity SaaS | Collaboration & real-time | Presence, conflict resolution, websocket scaling | realtime_collab_service.yaml, concurrency_control_notes.md | presence freshness latency, conflict resolution success rate |
| E-commerce | Dynamic catalog & checkout | Inventory consistency, cart integrity, promotion engine | cart_consistency.md, inventory_reservation_design.md | cart abandonment %, inventory oversell incidents |
| Streaming Media | High throughput & personalization | CDN cache warming, recommendation freshness | cdn_controller_design.md, personalization_pipeline.md | cache warm success %, rec freshness window |

## Drill Tactics

- 30 min capacity math per scenario (QPS, storage, replication lag estimates).
- 1 Mermaid diagram + ASCII fallback per design.
- Trade-off matrix: at least 5 forces (latency, consistency, cost, complexity, scalability).
- Failure mode enumeration (top 5) + mitigations.
- Evolution path (V1 → V2 → scale milestone).

## Daily Rotation (Weeks 11–12)

| Day | Morning Focus | Midday Mock | Evening Review |
|-----|---------------|------------|----------------|
| Mon | FAANG scale patterns | Rate limiter design | Trade-off justification notes |
| Tue | Coinbase consistency | Ledger + idempotency | Reconciliation strategy refinement |
| Wed | Rippling multi-tenancy | Feature flag targeting | Isolation boundary risks |
| Thu | Fintech reliability | Fraud streaming pipeline | Retry + DLQ tuning |
| Fri | Productivity SaaS realtime | Collaboration OT | Conflict resolution improvements |
| Sat | E-commerce integrity | Cart consistency design | Promotion engine constraints |
| Sun | Streaming personalization | Recommendation freshness | Caching invalidation summary |

## Usage

- Week 11–12: rotate by company focus daily.
- Maintain notes of gaps and follow-up designs.
- Track metrics improvements in a simple spreadsheet for quick recall.
