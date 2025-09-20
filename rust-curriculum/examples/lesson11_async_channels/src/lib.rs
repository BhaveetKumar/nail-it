use tokio::sync::mpsc;

pub async fn sum_worker(mut rx: mpsc::Receiver<i32>) -> i32 {
    let mut sum = 0;
    while let Some(v) = rx.recv().await {
        sum += v;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sums_values() {
        let (tx, rx) = mpsc::channel(8);
        let handle = tokio::spawn(sum_worker(rx));
        for v in [1, 2, 3, 4] { tx.send(v).await.unwrap(); }
        drop(tx);
        let res = handle.await.unwrap();
        assert_eq!(res, 10);
    }
}
