---
verified: true
---

# Lesson 24: HTTP with Actix Web

This lesson introduces Actix Web by implementing a minimal `/health` endpoint that returns JSON. We'll use `actix_web::test` utilities to verify responses.

## Key Concepts

- Handlers returning `impl Responder` with JSON via `web::Json`.
- Using `#[get("/health")]` routing macro and composing an `App` in tests.
- Testing handlers with `actix_web::test` and extracting response body bytes.

## Example

The example crate `examples/lesson24_actix` includes:

- `health` handler responding `{ "status": "ok" }`.
- A test harness using `init_service`, `TestRequest`, and `call_service`.

## Run the tests

- `cargo test -p lesson24_actix`

## Further Reading

- [Actix Web Guide](https://actix.rs/docs/)
- [`actix-web` crate](https://crates.io/crates/actix-web)
