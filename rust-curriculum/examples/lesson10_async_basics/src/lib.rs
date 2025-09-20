use tokio::time::{sleep, Duration};

pub async fn slow_add(a: i32, b: i32, ms: u64) -> i32 {
    sleep(Duration::from_millis(ms)).await;
    a + b
}

pub async fn fetch_both() -> (i32, i32) {
    let f1 = slow_add(1, 2, 50);
    let f2 = slow_add(3, 4, 10);
    let (a, b) = tokio::join!(f1, f2);
    (a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn join_works() {
        let (a, b) = fetch_both().await;
        assert_eq!((a, b), (3, 7));
    }
}
