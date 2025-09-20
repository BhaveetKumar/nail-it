# Final Validation Report

Generated: 2025-09-19

Summary of lesson verification status based on workspace tests and content review.

Verified Lessons

- lessons/01_intro.md — verified: true
- lessons/02_ownership.md — verified: true
- lessons/03_tooling.md — verified: true
- lessons/04_pattern_matching.md — verified: true
- lessons/05_collections_iterators.md — verified: true
- lessons/06_error_handling.md — verified: true
- lessons/07_traits_generics.md — verified: true
- lessons/08_trait_objects.md — verified: true
- lessons/09_macros_intro.md — verified: true
- lessons/10_async_basics.md — verified: true
- lessons/11_async_channels.md — verified: true
- lessons/12_async_select_timeouts.md — verified: true
- lessons/13_async_io.md — verified: true
- lessons/14_http_axum.md — verified: true
- lessons/15_tracing_observability.md — verified: true
- lessons/16_sled_kv.md — verified: true
- lessons/17_property_testing.md — verified: true
- lessons/18_unsafe_basics.md — verified: true
- lessons/19_ffi_basics.md — verified: true
- lessons/20_parallelism_rayon.md — verified: true
- lessons/21_serde_json.md — verified: true
- lessons/22_concurrency_crossbeam.md — verified: true
- lessons/23_serde_advanced.md — verified: true
- lessons/24_http_actix.md — verified: true
- lessons/25_e2e_service_axum.md — verified: true

Evidence

- cargo test — packages passed for lesson21_serde, lesson22_crossbeam, lesson23_serde_advanced, lesson24_actix, and lesson25_e2e_service; full workspace tests green.
- Frontmatter contains `verified: true` for the above lessons; markdown lint passes.

Timestamps

- sources.json updated on 2025-09-19 with crate versions and fetched_at fields.
- curriculum_index.json generated_at: 2025-09-19T00:00:00Z

Notes

- CI is configured to run fmt, clippy (-D warnings), build, test, and audit in `rust-curriculum`.
