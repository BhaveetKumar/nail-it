#[macro_export]
macro_rules! maplit {
    ( $( $k:expr => $v:expr ),* $(,)? ) => {{
        let mut m: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
        $( m.insert($k.to_string(), $v as i32); )*
        m
    }}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn works() {
        let m = maplit! {"a" => 1, "b" => 2};
        assert_eq!(m.get("a"), Some(&1));
        assert_eq!(m.get("b"), Some(&2));
    }
}
