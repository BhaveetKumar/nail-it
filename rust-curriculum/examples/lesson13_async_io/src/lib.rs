use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub async fn roundtrip(msg: &[u8]) -> Vec<u8> {
    let (mut a, mut b) = tokio::io::duplex(64);
    let m = msg.to_vec();
    let w = tokio::spawn(async move {
        a.write_all(&m).await.unwrap();
        a.shutdown().await.unwrap();
    });
    let mut buf = vec![0u8; msg.len()];
    b.read_exact(&mut buf).await.unwrap();
    w.await.unwrap();
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn duplex_roundtrip() {
        let got = roundtrip(b"hello").await;
        assert_eq!(got, b"hello");
    }
}
