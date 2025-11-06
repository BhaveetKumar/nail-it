# Content Mapping (Legacy -> New Taxonomy)

| Legacy Path | New Target Folder | Rationale / Notes |
|-------------|-------------------|-------------------|
| 01_fundamentals/algorithms/* | 00_foundation/core_cs/algorithms/ | Core CS retained (prune overly niche variants) |
| 01_fundamentals/data_structures/* | 00_foundation/core_cs/data_structures/ | Keep representative DS w/ complexity annotations |
| 01_fundamentals/programming/GOLANG_* | 05_go_specialization/ | Consolidate Go learning; remove duplication |
| 01_fundamentals/ADVANCED_MICROSERVICES_PATTERNS.md | 02_system_design/patterns/microservices/ | Moves to canonical pattern set |
| 02_system_design/patterns/patterns/* | 02_system_design/patterns/ | Flatten nested duplicates; unify naming |
| 02_system_design/case_studies/* | 02_system_design/architecture_case_studies/ | Keep only those with explicit trade-offs table |
| 03_backend_engineering/microservices/* | 02_system_design/patterns/microservices/ | Architectural patterns not language-specific |
| 03_backend_engineering/databases/* | 04_data_engineering/ | Data design, optimization, partitioning |
| 03_backend_engineering/grpc/* | 01_backend_core/async_messaging_basics/ or 05_go_specialization/ | gRPC intro near messaging; Go-specific parts in Go folder |
| 03_backend_engineering/message_queues/* | 01_backend_core/async_messaging_basics/ | Intro; advanced streaming moves to 02 deep_dives |
| 04_devops_infrastructure/monitoring/* | 03_platform_and_performance/observability/ | Consolidate metrics/tracing/logging |
| 04_devops_infrastructure/performance/* | 03_platform_and_performance/performance/ | Performance engineering centralized |
| 04_devops_infrastructure/security/* | 03_platform_and_performance/security/ | Security architecture + threat modeling |
| 05_ai_ml/* (non-arch) | (Optional Extension) | Out of critical path for backend 90-day; mark optional |
| 06_behavioral/* | 07_interview_execution/behavioral/ | Unified behavioral prep location |
| 07_company_specific/razorpay/* | 07_interview_execution/system_design_rounds/company_specific/ | Scenario-focused revision |
| 08_interview_prep/guides/* | 07_interview_execution/system_design_rounds/ & behavioral/ | Merge by category (system vs behavioral) |
| 09_curriculum/* | 09_curriculum_90_day/ | Replace with refined phased plan |

## Duplicate Candidates (To Audit)
- Microservices pattern docs (found in fundamentals + backend_engineering + system_design).
- Distributed systems guides across multiple folders.
- Performance optimization repeated in architecture + devops.

## Action Rules
1. Preserve newest version (latest timestamp) as canonical.
2. Insert front matter before moving.
3. Add redirect note at old path if retained temporarily.
4. Record removed files with hash in `catalog.json` (future automation).

## Pending Audit Tags
- [ ] Microservices duplicates
- [ ] Event sourcing vs CQRS hybrids
- [ ] Rate limiting variations
- [ ] Caching patterns proliferation
