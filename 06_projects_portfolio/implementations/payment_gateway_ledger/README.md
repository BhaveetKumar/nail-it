---
# Auto-generated front matter
Title: Readme
LastUpdated: 2025-11-06T20:45:58.373237
Tags: []
Status: draft
---

# Payment Gateway & Ledger (Prompt 3)

Minimal skeleton showing idempotent payment ingestion + ledger append with retry and reconciliation hooks.

## Core Concepts

- Idempotency via request key store
- Two-phase: auth -> capture simulation
- Append-only ledger, eventual reconciliation window

## Run

```bash
cd 06_projects_portfolio/implementations/payment_gateway_ledger
go run ./...
```

## Extend

- Swap in Postgres + serializable isolation for ledger writes
- Add outbox table for event sourcing to Kafka

## Testing

- Unit: idempotent replay
- Failure injection: partial auth vs capture
