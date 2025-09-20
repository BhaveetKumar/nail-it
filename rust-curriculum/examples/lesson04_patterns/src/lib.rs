pub fn eval(expr: &str) -> Option<i64> {
    let parts: Vec<_> = expr.split('+').map(|s| s.trim()).collect();
    match parts.as_slice() {
        [a, b] => {
            let (Ok(a), Ok(b)) = (a.parse::<i64>(), b.parse::<i64>()) else {
                return None;
            };
            Some(a + b)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ok() {
        assert_eq!(eval("2 + 3"), Some(5));
    }
    #[test]
    fn bad() {
        assert_eq!(eval("x + 3"), None);
    }
}
