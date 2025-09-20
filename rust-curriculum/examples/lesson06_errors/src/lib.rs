use anyhow::{Context, Result};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CsvSumError {
    #[error("empty input")]
    Empty,
}

pub fn csv_sum(line: &str) -> Result<i64> {
    if line.trim().is_empty() {
        return Err(CsvSumError::Empty.into());
    }
    line.split(',')
        .map(|s| s.trim().parse::<i64>().context("parse int"))
        .try_fold(0i64, |acc, r| r.map(|v| acc + v))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ok() {
        assert_eq!(csv_sum("1, 2, 3").unwrap(), 6);
    }
    #[test]
    fn empty() {
        assert!(csv_sum("").is_err());
    }
    #[test]
    fn bad() {
        assert!(csv_sum("a,2").is_err());
    }
}
