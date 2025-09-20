use anyhow::{anyhow, Result};
use tokio::time::{sleep, timeout, Duration};

pub async fn maybe_after(ms: u64, ok: bool) -> Result<&'static str> {
    sleep(Duration::from_millis(ms)).await;
    if ok { Ok("ok") } else { Err(anyhow!("bad")) }
}

pub async fn first_ok<A, B>(a: A, b: B) -> Result<&'static str>
where
    A: std::future::Future<Output = Result<&'static str>>,
    B: std::future::Future<Output = Result<&'static str>>,
{
    tokio::pin!(a);
    tokio::pin!(b);
    tokio::select! {
        ra = &mut a => {
            match ra {
                Ok(v) => Ok(v),
                Err(_) => b.await,
            }
        },
        rb = &mut b => {
            match rb {
                Ok(v) => Ok(v),
                Err(_) => a.await,
            }
        },
    }
}

pub async fn with_timeout<F, T>(dur: Duration, fut: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    Ok(timeout(dur, fut).await??)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn picks_first_ok() {
        let r = first_ok(maybe_after(5, true), maybe_after(1, false)).await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn times_out() {
        let r = with_timeout(Duration::from_millis(1), maybe_after(50, true)).await;
        assert!(r.is_err());
    }
}
