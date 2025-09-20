---
updated_at: 2025-09-19T00:00:00Z
---

# Async Patterns & Pitfalls

## Common Pitfalls

- Blocking in async: avoid `std::thread::sleep`; use `tokio::time::sleep`.
- Holding locks across `.await`: can deadlock. Drop guards before awaiting.
- Long-lived borrows across `.await`: move owned data or clone small pieces.
- `select!` cancellation: spawned tasks may keep running; use `AbortHandle`.
- CPU-bound work on runtime: offload to `spawn_blocking` or a dedicated pool.

## Patterns

- Cancellation:

```rust
use tokio::{task::JoinHandle, sync::oneshot};
fn cancellable<F, T>(fut: F) -> (JoinHandle<T>, oneshot::Sender<()>)
where
    F: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let (tx, mut rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        tokio::select! {
            _ = &mut rx => panic!("cancelled"),
            res = fut => res,
        }
    });
    (handle, tx)
}
```

- Timeouts:

```rust
use std::time::Duration;
use tokio::time::timeout;
async fn with_timeout<F, T>(dur: Duration, fut: F) -> anyhow::Result<T>
where
    F: std::future::Future<Output = T>,
{
    Ok(timeout(dur, fut).await?)
}
```

- Bounded concurrency:

```rust
use futures::stream::{self, StreamExt};
async fn map_concurrent<I, F, Fut, T>(items: Vec<I>, limit: usize, f: F) -> Vec<T>
where
    F: Fn(I) -> Fut + Send + Sync + Copy + 'static,
    Fut: std::future::Future<Output = T> + Send,
    I: Send + 'static,
    T: Send + 'static,
{
    stream::iter(items)
        .map(|i| f(i))
        .buffer_unordered(limit)
        .collect()
        .await
}
```

## Further Reading

- Tokio: Time, Sync, Tasks — fetched_at: 2025-09-19T00:00:00Z
- Rust Async Book — fetched_at: 2025-09-19T00:00:00Z
