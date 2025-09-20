use tracing::{info, span, Level};

pub fn init_for_tests() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();
}

pub fn compute(x: i32) -> i32 {
    let s = span!(Level::INFO, "compute", input = x);
    let _e = s.enter();
    let y = x + 1;
    info!(result = y, "computed");
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_logs() {
        init_for_tests();
        let r = compute(41);
        assert_eq!(r, 42);
    }
}
