---
updated_at: 2025-09-19T00:00:00Z
---

# Macro Quick Reference

- Matchers: `tt`, `ident`, `path`, `expr`, `ty`, `pat`, `stmt`, `block`, `meta`, `lifetime`, `literal`.
- Repetition: `$( ... )*`, `+`, `?` with separators: `$( $x:expr ),*`.
- Trailing comma: use `$(,)?` to accept optional trailing separators.
- Hygiene: macros have their own scopes; prefer `$crate::` to refer to the current crate.
- Export: `#[macro_export]` to make available to dependents; consider a prelude module.
- Debugging: expand with `cargo expand` (install `cargo install cargo-expand`).
- Avoid surprises: limit `tt` when `expr`/`ident` suffices; document expansion.

## Examples

- Vec builder:

```rust
#[macro_export]
macro_rules! vec_of {
    ( $( $x:expr ),* $(,)? ) => {
        vec![ $( $x ),* ]
    }
}
```

- Map builder with types:

```rust
#[macro_export]
macro_rules! map_of {
    ( $( $k:expr => $v:expr ),* $(,)? ) => {{
        let mut m = std::collections::HashMap::new();
        $( m.insert($k, $v); )*
        m
    }}
}
```

## Tips

- Keep rules specific and minimal; add more arms when needed.
- Use `:tt` sparingly; prefer structured matchers.
- Consider functions/generics before macros for readability.
- Provide tests for macro behavior and error cases.
