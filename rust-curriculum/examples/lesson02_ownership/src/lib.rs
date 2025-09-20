pub fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b' ' {
            return &s[..i];
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty() {
        assert_eq!(first_word(""), "");
    }
    #[test]
    fn one() {
        assert_eq!(first_word("abc"), "abc");
    }
    #[test]
    fn two() {
        assert_eq!(first_word("abc def"), "abc");
    }
}
